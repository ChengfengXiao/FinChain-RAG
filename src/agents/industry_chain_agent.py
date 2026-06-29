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


def build_financial_quality_snapshots(sources: list[AShareSource]) -> list[dict[str, Any]]:
    snapshots = []
    for source in sources:
        if source.status == "ok" and source.source == "financial_quality" and isinstance(source.data, dict):
            snapshots.append(source.data)
    return snapshots


def latest_value(snapshot: dict[str, Any], field: str) -> Any:
    latest = snapshot.get("latest_period") or {}
    return latest.get(field)


def positive_count(rows: list[dict[str, Any]], field: str) -> int:
    return sum(1 for row in rows if isinstance(row.get(field), (int, float)) and row.get(field) > 0)


def growth_rate(rows: list[dict[str, Any]], field: str) -> float | None:
    valid = [row.get(field) for row in rows if isinstance(row.get(field), (int, float))]
    if len(valid) < 2 or valid[-1] in {0, None}:
        return None
    oldest = valid[-1]
    latest = valid[0]
    if not isinstance(oldest, (int, float)) or oldest == 0:
        return None
    return latest / oldest - 1


def score_financial_quality(snapshot: dict[str, Any]) -> dict[str, Any]:
    annual = snapshot.get("annual_periods", [])
    latest = annual[0] if annual else snapshot.get("latest_period", {})
    score = 0
    details: list[str] = []

    deduct_ratio = latest.get("deduct_net_profit_ratio")
    if isinstance(deduct_ratio, (int, float)):
        if deduct_ratio >= 0.9:
            score += 12
            details.append("扣非净利润与归母净利润匹配度高")
        elif deduct_ratio >= 0.7:
            score += 8
            details.append("扣非净利润与归母净利润基本匹配")
        else:
            score += 3
            details.append("扣非净利润占比偏低，需核查非经常性损益")
    else:
        details.append("扣非净利润字段缺失，利润质量扣分")

    ocf_ratio = latest.get("ocf_net_profit_ratio")
    if isinstance(ocf_ratio, (int, float)):
        if ocf_ratio >= 1:
            score += 15
            details.append("经营现金流覆盖净利润")
        elif ocf_ratio >= 0.6:
            score += 9
            details.append("经营现金流部分覆盖净利润")
        else:
            score += 3
            details.append("经营现金流覆盖净利润不足")
    else:
        details.append("经营现金流覆盖率缺失")

    if positive_count(annual, "free_cashflow") >= 4:
        score += 12
        details.append("近5年多数年份自由现金流为正")
    elif positive_count(annual, "free_cashflow") >= 2:
        score += 7
        details.append("近5年部分年份自由现金流为正")
    else:
        score += 2
        details.append("自由现金流稳定性不足")

    revenue_growth = growth_rate(annual, "revenue")
    deduct_growth = growth_rate(annual, "deduct_net_profit")
    ocf_growth = growth_rate(annual, "operating_cashflow")
    growth_hits = sum(1 for item in [revenue_growth, deduct_growth, ocf_growth] if isinstance(item, (int, float)) and item > 0)
    score += {3: 15, 2: 10, 1: 5}.get(growth_hits, 1)
    details.append(f"近5年营收/扣非/经营现金流增长同步项：{growth_hits}/3")

    debt_ratio = latest.get("debt_asset_ratio")
    cash_short = latest.get("cash_short_debt_ratio")
    interest_coverage = latest.get("interest_coverage")
    if isinstance(debt_ratio, (int, float)) and debt_ratio <= 0.55:
        score += 8
    elif isinstance(debt_ratio, (int, float)) and debt_ratio <= 0.7:
        score += 5
    else:
        score += 1
    if isinstance(cash_short, (int, float)) and cash_short >= 1:
        score += 5
    elif cash_short is None:
        details.append("现金短债比字段不完整")
    if isinstance(interest_coverage, (int, float)) and interest_coverage >= 3:
        score += 5
    elif interest_coverage is None:
        details.append("利息保障倍数字段不完整")

    receivable_ratio = latest.get("receivable_revenue_ratio")
    inventory_ratio = latest.get("inventory_revenue_ratio")
    goodwill_ratio = latest.get("goodwill_assets_ratio")
    if isinstance(receivable_ratio, (int, float)) and receivable_ratio <= 0.35:
        score += 6
    elif isinstance(receivable_ratio, (int, float)) and receivable_ratio <= 0.6:
        score += 3
    if isinstance(inventory_ratio, (int, float)) and inventory_ratio <= 0.25:
        score += 5
    elif isinstance(inventory_ratio, (int, float)) and inventory_ratio <= 0.5:
        score += 2
    if goodwill_ratio is None or goodwill_ratio <= 0.05:
        score += 5
    elif goodwill_ratio <= 0.15:
        score += 2

    score += 10  # 商业模式、行业地位、护城河、估值需要 LLM 结合文本和估值字段进一步修正。
    score = max(0, min(int(round(score)), 100))
    if score >= 85:
        label = "顶级公司"
    elif score >= 70:
        label = "优秀公司"
    elif score >= 55:
        label = "普通公司"
    elif score >= 40:
        label = "高风险公司"
    else:
        label = "应回避公司"
    return {"score": score, "label": label, "details": details}


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
            relation = "同层/同主题公司线索"
            if row_layer < target_layer:
                relation = "可能一层上游公司线索"
            elif row_layer > target_layer:
                relation = "可能一层下游/销售方向公司线索"
            rows.append(
                {
                    "company_name": row.get("company_name", ""),
                    "ticker": code,
                    "segment": row.get("segment", ""),
                    "sub_segment": row.get("sub_segment", ""),
                    "relation": relation,
                    "relation_quality": "industry_mapping_only",
                    "evidence": "companies.jsonl 结构化产业链映射，非财报披露的已验证客户/供应商",
                    "leader_score": row.get("leader_score", 0),
                }
            )
    return sorted(rows, key=lambda row: row.get("leader_score") or 0, reverse=True)[:limit]


def format_yuan(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "未披露"
    if abs(number) >= 100000000:
        return f"{number / 100000000:.2f}亿元"
    if abs(number) >= 10000:
        return f"{number / 10000:.2f}万元"
    return f"{number:.2f}元"


def format_ratio(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "未披露"
    return f"{number * 100:.2f}%"


def format_mix_summary(rows: list[dict[str, Any]], amount_key: str, ratio_key: str) -> list[dict[str, Any]]:
    summary = []
    for row in rows[:6]:
        summary.append(
            {
                "item_name": row.get("item_name") or "未命名项目",
                "report_date": row.get("report_date") or "",
                "amount": format_yuan(row.get(amount_key)),
                "ratio": format_ratio(row.get(ratio_key)),
                "gross_margin": format_ratio(row.get("gross_margin")),
            }
        )
    return summary


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
        company_title = json.dumps(
            {
                "code": code,
                "name": name,
                "metrics": item.get("metrics", {}),
                "business_model": item.get("business_model", {}),
                "composition_date": item.get("composition_date", ""),
                "revenue_mix": format_mix_summary(item.get("revenue_mix", []), "revenue_yuan", "revenue_ratio"),
                "cost_mix": format_mix_summary(item.get("cost_mix", []), "cost_yuan", "cost_ratio"),
                "note": "收入和成本来自主营构成，只作为财务构成说明；公开抓取数据未披露具体客户或供应商公司名时，不生成客户/供应商节点。",
            },
            ensure_ascii=False,
        )
        add_node(company_id, f"{name}\n{code}", "company", company_title, 22)

        for related in related_companies_for_target(code, company_rows):
            related_id = f"related:{related['ticker']}"
            add_node(related_id, f"{related['company_name']}\n{related['ticker']}", "related", json.dumps(related, ensure_ascii=False), 14)
            if "上游" in related["relation"]:
                add_edge(related_id, company_id, related["relation"], related["evidence"])
            else:
                add_edge(company_id, related_id, related["relation"], related["evidence"])

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "counterparty_policy": "only_draw_disclosed_or_mapped_companies",
        "disclosed_counterparties": [],
        "note": "图谱只连公司节点。若公开财报/抓取数据未披露具体客户或供应商公司名，收入和成本只在主营构成里展示具体金额、占比、毛利率，不画成关系节点。",
    }


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
        if mode not in {"company_quality", "company_ops"}:
            mode = "company_quality"

        chat_config = resolve_chat_config(provider=provider, model=model)
        client = create_chat_client(provider=chat_config.provider)
        companies = load_companies()

        data_client = AShareDataClient()
        codes, online_sources = data_client.collect_company_ops(question, limit=1)
        matched_companies = companies[companies["ticker"].astype(str).isin(codes)] if codes else filter_companies(question, companies, limit=1)
        snapshots = build_operating_snapshots(online_sources)
        financial_quality = build_financial_quality_snapshots(online_sources)
        quality_score = score_financial_quality(financial_quality[0]) if financial_quality else {"score": 0, "label": "数据不足", "details": ["未抓取到财务质量数据"]}
        run_date = current_run_date()
        user_prompt = build_company_ops_prompt(
            question=question,
            run_date=run_date,
            online_context=format_online_context(online_sources),
            financial_context=json.dumps(
                {
                    "targets": codes,
                    "operating_snapshots": snapshots,
                    "financial_quality": financial_quality,
                    "preliminary_score": quality_score,
                    "structured_company_mapping": format_company_context(matched_companies),
                },
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
            "financial_quality": financial_quality,
            "quality_score": quality_score,
        }
