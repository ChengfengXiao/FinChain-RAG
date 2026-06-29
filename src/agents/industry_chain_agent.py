from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from src.data.ashare_data import AShareDataClient, AShareSource, normalize_code
from src.prompts.report_prompt import (
    ONLINE_RESEARCH_SYSTEM_PROMPT,
    REPORT_SYSTEM_PROMPT,
    build_online_research_prompt,
    build_report_prompt,
)
from src.settings import COMPANIES_PATH
from src.llm.providers import create_chat_client, resolve_chat_config


@dataclass
class RetrievedChunkLike:
    document: str
    metadata: dict
    distance: float | None = None


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


def format_retrieved_context(chunks: list[RetrievedChunkLike]) -> str:
    if not chunks:
        return "未检索到相关资料。"
    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        chunk_index = chunk.metadata.get("chunk_index", "unknown")
        lines.append(f"[{idx}] source={source}, chunk_index={chunk_index}\n{chunk.document}")
    return "\n\n".join(lines)


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
        item = grouped.setdefault(source.code, {"code": source.code, "name": "", "metrics": {}, "concepts": [], "announcements": [], "news": []})
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

    query_id = "query"
    add_node(query_id, question[:28] + ("..." if len(question) > 28 else ""), "query", question, 26)

    company_lookup = {
        str(row.get("ticker", "")): {
            "name": str(row.get("company_name", "")),
            "segment": str(row.get("segment", "")),
            "sub_segment": str(row.get("sub_segment", "")),
            "leader_score": row.get("leader_score", ""),
        }
        for _, row in company_rows.iterrows()
    }

    search_blocks: list[dict[str, Any]] = []
    for source in sources:
        if source.source == "eastmoney_search" and isinstance(source.data, dict):
            search_blocks.extend(source.data.get("selected_blocks", []))

    for block in search_blocks[:3]:
        block_id = f"block:{block.get('code')}"
        add_node(block_id, block.get("name", block.get("code", "板块")), "industry", json.dumps(block, ensure_ascii=False), 24)
        add_edge(query_id, block_id, "匹配板块", "来自东财关键词搜索")

    snapshots = build_operating_snapshots(sources)
    for item in snapshots:
        code = item["code"]
        name = item.get("name") or company_lookup.get(code, {}).get("name") or code
        company_id = f"company:{code}"
        metrics = item.get("metrics", {})
        company_title = json.dumps({"code": code, "name": name, "metrics": metrics}, ensure_ascii=False)
        add_node(company_id, f"{name}\n{code}", "company", company_title, 22)
        add_edge(query_id, company_id, "检索标的", "由代码、公司名或板块成分映射")

        if code in company_lookup:
            segment = company_lookup[code]["segment"]
            sub_segment = company_lookup[code]["sub_segment"]
            segment_id = f"segment:{segment}"
            add_node(segment_id, segment, "segment", sub_segment, 18)
            add_edge(segment_id, company_id, "公司映射", sub_segment)

        industry = metrics.get("industry")
        if industry:
            industry_id = f"industry:{industry}"
            add_node(industry_id, str(industry), "industry", "东财个股基本面行业字段", 18)
            add_edge(company_id, industry_id, "所属行业", "东财个股基本面")

        for concept in item.get("concepts", [])[:6]:
            concept_id = f"concept:{concept}"
            add_node(concept_id, str(concept), "concept", "东财概念/行业板块", 14)
            add_edge(company_id, concept_id, "概念关联", "东财概念/行业板块")

        if item.get("announcements"):
            ann_id = f"ann:{code}"
            add_node(ann_id, "近期公告", "evidence", json.dumps(item["announcements"], ensure_ascii=False), 13)
            add_edge(company_id, ann_id, "公告证据", "巨潮公告")

        if item.get("news"):
            news_id = f"news:{code}"
            add_node(news_id, "近期新闻", "evidence", json.dumps(item["news"], ensure_ascii=False), 13)
            add_edge(company_id, news_id, "新闻线索", "东财新闻")

        fund_date = metrics.get("fund_latest_date")
        if fund_date:
            fund_id = f"fund:{code}"
            add_node(fund_id, f"资金流\n{fund_date}", "evidence", json.dumps(metrics, ensure_ascii=False), 13)
            add_edge(company_id, fund_id, "资金流", "东财120日资金流摘要")

    return {"nodes": list(nodes.values()), "edges": edges}


def current_run_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


class IndustryChainAgent:
    def __init__(self) -> None:
        self._retriever: Any | None = None

    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            from src.retriever.retriever import ChromaRetriever

            self._retriever = ChromaRetriever()
        return self._retriever

    def answer(
        self,
        question: str,
        top_k: int = 5,
        provider: str | None = None,
        model: str | None = None,
        research_mode: str = "local_rag",
        online_limit: int = 2,
    ) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("question cannot be empty")

        mode = research_mode.lower().strip()
        if mode not in {"local_rag", "a_stock_online", "bottleneck_hunter", "serenity"}:
            raise ValueError("research_mode must be one of: local_rag, a_stock_online, bottleneck_hunter, serenity")

        chat_config = resolve_chat_config(provider=provider, model=model)
        client = create_chat_client(provider=chat_config.provider)
        companies = load_companies()

        if mode != "local_rag":
            matched_companies = filter_companies(question, companies, limit=online_limit)
            data_client = AShareDataClient()
            codes = extract_codes_from_question(question, matched_companies if not matched_companies.empty else companies, limit=online_limit)
            search_sources: list[AShareSource] = []
            if not codes:
                codes, search_sources, _ = data_client.resolve_targets(question, limit=online_limit)
            online_sources = [*search_sources, *data_client.collect(codes)]
            snapshots = build_operating_snapshots(online_sources)
            graph = build_relationship_graph(question, online_sources, matched_companies)
            run_date = current_run_date()
            user_prompt = build_online_research_prompt(
                question=question,
                research_mode=mode,
                run_date=run_date,
                online_context=format_online_context(online_sources),
                company_context=format_company_context(matched_companies),
            )
            response = client.chat.completions.create(
                model=chat_config.model,
                messages=[
                    {"role": "system", "content": ONLINE_RESEARCH_SYSTEM_PROMPT},
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

        chunks = self.retriever.retrieve(question, top_k=top_k)
        filtered_companies = filter_companies(question, companies)
        user_prompt = build_report_prompt(
            question=question,
            retrieved_context=format_retrieved_context(chunks),
            company_context=format_company_context(filtered_companies),
        )

        response = client.chat.completions.create(
            model=chat_config.model,
            messages=[
                {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content or ""
        sources = [
            {
                "source": chunk.metadata.get("source"),
                "chunk_index": chunk.metadata.get("chunk_index"),
                "theme": chunk.metadata.get("theme"),
                "distance": chunk.distance,
            }
            for chunk in chunks
        ]
        return {
            "answer": answer,
            "sources": sources,
            "provider": chat_config.provider,
            "model": chat_config.model,
            "research_mode": mode,
            "run_date": current_run_date(),
            "targets": [],
            "operating_snapshots": [],
            "graph": {"nodes": [], "edges": []},
        }
