from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

import pandas as pd
from src.data.ashare_data import AShareDataClient, AShareSource, normalize_code
from src.prompts.report_prompt import (
    COMPANY_OPS_SYSTEM_PROMPT,
    build_company_ops_prompt,
)
from src.settings import COMPANIES_PATH
from src.llm.providers import create_chat_client, resolve_chat_config


def load_companies() -> pd.DataFrame:
    if not COMPANIES_PATH.exists():
        raise RuntimeError(f"Company data file not found: {COMPANIES_PATH}")
    records = [json.loads(line) for line in COMPANIES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(records)


def filter_companies(question: str, companies: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    if companies.empty:
        return companies

    query = question.lower().strip()
    searchable_fields = ["company_name", "ticker", "segment", "sub_segment", "business_description", "relevance_reason"]

    def score_row(row: pd.Series) -> int:
        text = " ".join(str(row.get(field, "")).lower() for field in searchable_fields)
        score = 0
        for field in searchable_fields:
            value = str(row.get(field, "")).lower()
            if value and value in query:
                score += 3
        for keyword in ["上游", "中游", "下游", "冷板", "cdu", "泵", "阀", "管路", "密封", "温控", "机柜", "公司", "龙头"]:
            if keyword in query and keyword in text:
                score += 1
        return score

    scores = companies.apply(score_row, axis=1)
    filtered = companies[scores > 0] if (scores > 0).any() else companies
    filtered = filtered.assign(_match_score=scores.loc[filtered.index])
    return filtered.sort_values(["_match_score", "leader_score"], ascending=[False, False]).drop(columns=["_match_score"]).head(limit)


def format_company_context(companies: pd.DataFrame) -> str:
    if companies.empty:
        return "未找到结构化公司数据。"
    fields = [
        "company_name",
        "ticker",
        "segment",
        "sub_segment",
        "business_description",
        "relevance_reason",
        "leader_score",
        "risk_note",
    ]
    return companies[fields].to_json(orient="records", force_ascii=False, indent=2)


def extract_codes_from_question(question: str, companies: pd.DataFrame, limit: int = 8) -> list[str]:
    codes: list[str] = []
    for match in re.finditer(r"(?<!\d)(?:SH|SZ|BJ)?(\d{6})(?:\.(?:SH|SZ|BJ))?(?!\d)", question, re.IGNORECASE):
        code = normalize_code(match.group(0))
        if code and code not in codes:
            codes.append(code)

    for _, row in companies.iterrows():
        name = str(row.get("company_name", ""))
        ticker = str(row.get("ticker", ""))
        if name and name in question and ticker and ticker not in codes:
            codes.append(ticker)

    if not codes:
        for ticker in filter_companies(question, companies, limit=limit)["ticker"].astype(str).tolist():
            if ticker not in codes:
                codes.append(ticker)

    return codes[:limit]


def format_online_context(sources: list[AShareSource]) -> str:
    if not sources:
        return "未抓取到在线 A 股数据。"
    records = [
        {
            "source": source.source,
            "code": source.code,
            "title": source.title,
            "status": source.status,
            "fetched_at": source.fetched_at,
            "data": source.data,
        }
        for source in sources
    ]
    return json.dumps(records, ensure_ascii=False, indent=2)


def format_source_rows(sources: list[AShareSource]) -> list[dict[str, Any]]:
    return [
        {
            "source": source.source,
            "code": source.code,
            "title": source.title,
            "status": source.status,
            "fetched_at": source.fetched_at,
        }
        for source in sources
    ]


def build_operating_snapshots(sources: list[AShareSource]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for source in sources:
        if source.status != "ok" or not source.code:
            continue
        item = grouped.setdefault(
            source.code,
            {
                "code": source.code,
                "name": "",
                "metrics": {},
                "business_scope": "",
                "business_review": "",
                "composition_date": "",
                "revenue_mix": [],
                "cost_mix": [],
                "business_model": {},
                "concepts": [],
                "announcements": [],
                "news": [],
            },
        )
        data = source.data
        if source.source == "tencent_quote" and isinstance(data, dict):
            item["name"] = data.get("name") or item["name"]
            item["metrics"].update(
                {
                    "price": data.get("price"),
                    "change_pct": data.get("change_pct"),
                    "pe_ttm": data.get("pe_ttm"),
                    "pb": data.get("pb"),
                    "mcap_yi": data.get("mcap_yi"),
                    "turnover_pct": data.get("turnover_pct"),
                }
            )
        elif source.source == "eastmoney_stock_info" and isinstance(data, dict):
            item["name"] = data.get("name") or item["name"]
            item["metrics"].update(
                {
                    "industry": data.get("industry"),
                    "list_date": data.get("list_date"),
                    "mcap_yuan": data.get("mcap_yuan"),
                    "float_mcap_yuan": data.get("float_mcap_yuan"),
                }
            )
        elif source.source == "eastmoney_business_analysis" and isinstance(data, dict):
            composition = data.get("composition", [])
            product_rows = [row for row in composition if row.get("type") == "按产品"]
            revenue_mix = sorted(product_rows or composition, key=lambda row: row.get("revenue_ratio") or 0, reverse=True)[:6]
            cost_mix = sorted(product_rows or composition, key=lambda row: row.get("cost_ratio") or 0, reverse=True)[:6]
            evidence_text = " ".join(
                [
                    data.get("business_scope", ""),
                    " ".join(str(row.get("item_name", "")) for row in revenue_mix),
                    item["metrics"].get("industry", "") or "",
                ]
            )
            item.update(
                {
                    "business_scope": data.get("business_scope", ""),
                    "business_review": data.get("business_review", ""),
                    "composition_date": data.get("latest_report_date", ""),
                    "revenue_mix": revenue_mix,
                    "cost_mix": cost_mix,
                    "business_model": infer_business_model(evidence_text),
                }
            )
        elif source.source == "eastmoney_concept_blocks" and isinstance(data, dict):
            item["concepts"] = data.get("concept_tags", [])[:12]
        elif source.source == "stock_fund_flow_120d" and isinstance(data, dict):
            item["metrics"].update(
                {
                    "fund_latest_date": data.get("latest_date"),
                    "main_net_5d_yuan": data.get("main_net_5d_yuan"),
                    "main_net_20d_yuan": data.get("main_net_20d_yuan"),
                    "super_net_5d_yuan": data.get("super_net_5d_yuan"),
                }
            )
        elif source.source == "cninfo_announcements" and isinstance(data, list):
            item["announcements"] = data[:3]
        elif source.source == "eastmoney_stock_news" and isinstance(data, list):
            item["news"] = data[:3]
    return list(grouped.values())


def infer_business_model(text: str) -> dict[str, str]:
    to_b_keywords = ["设备", "工业", "数据中心", "机房", "通信", "客户", "工程", "系统", "企业", "电力", "汽车", "能源", "服务器"]
    to_c_keywords = ["消费者", "零售", "门店", "个人", "食品", "饮料", "服装", "家居", "旅游", "游戏", "会员"]
    b_hits = [word for word in to_b_keywords if word in text]
    c_hits = [word for word in to_c_keywords if word in text]
    if b_hits and not c_hits:
        return {"model": "ToB", "evidence": "、".join(b_hits[:5]), "confidence": "中"}
    if c_hits and not b_hits:
        return {"model": "ToC", "evidence": "、".join(c_hits[:5]), "confidence": "中"}
    if b_hits and c_hits:
        return {"model": "ToB/ToC 混合", "evidence": "、".join((b_hits + c_hits)[:6]), "confidence": "低"}
    return {"model": "不确定", "evidence": "主营范围和主营构成未提供足够业务对象线索", "confidence": "低"}


def related_companies_for_target(target_code: str, companies: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    if companies.empty:
        return []
    target = companies[companies["ticker"].astype(str) == str(target_code)]
    if target.empty:
        return []

    target_segment = str(target.iloc[0].get("segment", ""))

    def layer(segment: str) -> int:
        if "上游" in segment:
            return 1
        if "中游" in segment:
            return 2
        if "下游" in segment:
            return 3
        return 2

    target_layer = layer(target_segment)
    rows = []
    for _, row in companies.iterrows():
        code = str(row.get("ticker", ""))
        if code == str(target_code):
            continue
        row_layer = layer(str(row.get("segment", "")))
        if abs(row_layer - target_layer) <= 1:
            relation = "同层/同主题"
            if row_layer < target_layer:
                relation = "一层上游线索"
            elif row_layer > target_layer:
                relation = "一层下游线索"
            rows.append(
                {
                    "company_name": row.get("company_name", ""),
                    "ticker": code,
                    "segment": row.get("segment", ""),
                    "sub_segment": row.get("sub_segment", ""),
                    "relation": relation,
                    "evidence": "companies.jsonl 结构化产业链映射，非已验证客户/供应商",
                    "leader_score": row.get("leader_score", 0),
                }
            )
    return sorted(rows, key=lambda row: row.get("leader_score") or 0, reverse=True)[:limit]


def build_relationship_graph(question: str, sources: list[AShareSource], company_rows: pd.DataFrame) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def add_node(node_id: str, label: str, group: str, title: str = "", value: int = 16) -> None:
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "label": label, "group": group, "title": title or label, "value": value}

    def add_edge(source: str, target: str, label: str, title: str = "") -> None:
        if source in nodes and target in nodes:
            edge = {"from": source, "to": target, "label": label, "title": title or label}
            if edge not in edges:
                edges.append(edge)

    snapshots = build_operating_snapshots(sources)
    for item in snapshots:
        code = item["code"]
        if not code:
            continue
        name = item.get("name") or code
        company_id = f"company:{code}"
        company_title = json.dumps({"code": code, "name": name, "metrics": item.get("metrics", {})}, ensure_ascii=False)
        add_node(company_id, f"{name}\n{code}", "company", company_title, 22)

        for row in item.get("revenue_mix", [])[:5]:
            label = row.get("item_name") or "收入来源"
            node_id = f"revenue:{code}:{label}"
            title = json.dumps(row, ensure_ascii=False)
            add_node(node_id, f"收入\n{label}", "revenue", title, 16)
            add_edge(company_id, node_id, f"收入来源 {row.get('revenue_ratio', 0):.1%}", "东财F10主营构成")

        for row in item.get("cost_mix", [])[:5]:
            label = row.get("item_name") or "成本去向"
            node_id = f"cost:{code}:{label}"
            title = json.dumps(row, ensure_ascii=False)
            add_node(node_id, f"成本\n{label}", "cost", title, 16)
            add_edge(node_id, company_id, f"成本构成 {row.get('cost_ratio', 0):.1%}", "东财F10主营构成；未等同具体供应商")

        model = item.get("business_model", {})
        model_id = f"model:{code}:{model.get('model', '不确定')}"
        add_node(model_id, model.get("model", "ToB/ToC 不确定"), "model", json.dumps(model, ensure_ascii=False), 18)
        add_edge(company_id, model_id, "业务对象", model.get("evidence", ""))

        for related in related_companies_for_target(code, company_rows):
            related_id = f"related:{related['ticker']}"
            add_node(related_id, f"{related['company_name']}\n{related['ticker']}", "related", json.dumps(related, ensure_ascii=False), 14)
            if "上游" in related["relation"]:
                add_edge(related_id, company_id, related["relation"], related["evidence"])
            else:
                add_edge(company_id, related_id, related["relation"], related["evidence"])

    return {"nodes": list(nodes.values()), "edges": edges}


def current_run_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


class IndustryChainAgent:
    def __init__(self) -> None:
        pass

    def answer(
        self,
        question: str,
        top_k: int = 5,
        provider: str | None = None,
        model: str | None = None,
        research_mode: str = "company_ops",
        online_limit: int = 1,
    ) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("question cannot be empty")

        mode = research_mode.lower().strip()
        if mode != "company_ops":
            mode = "company_ops"

        chat_config = resolve_chat_config(provider=provider, model=model)
        client = create_chat_client(provider=chat_config.provider)
        companies = load_companies()

        data_client = AShareDataClient()
        codes, online_sources = data_client.collect_company_ops(question, limit=1)
        matched_companies = companies[companies["ticker"].astype(str).isin(codes)] if codes else filter_companies(question, companies, limit=1)
        snapshots = build_operating_snapshots(online_sources)
        graph = build_relationship_graph(question, online_sources, companies)
        run_date = current_run_date()
        user_prompt = build_company_ops_prompt(
            question=question,
            run_date=run_date,
            online_context=format_online_context(online_sources),
            graph_context=json.dumps(
                {"targets": codes, "snapshots": snapshots, "graph": graph, "related_company_mapping": format_company_context(matched_companies)},
                ensure_ascii=False,
                indent=2,
            ),
        )
        response = client.chat.completions.create(
            model=chat_config.model,
            messages=[
                {"role": "system", "content": COMPANY_OPS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content or ""
        return {
            "answer": answer,
            "sources": format_source_rows(online_sources),
            "provider": chat_config.provider,
            "model": chat_config.model,
            "research_mode": mode,
            "run_date": run_date,
            "targets": codes,
            "operating_snapshots": snapshots,
            "graph": graph,
        }
