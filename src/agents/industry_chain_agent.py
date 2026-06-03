from __future__ import annotations

import json
from typing import Any

import pandas as pd
from src.prompts.report_prompt import REPORT_SYSTEM_PROMPT, build_report_prompt
from src.retriever.retriever import ChromaRetriever, RetrievedChunk
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


def format_retrieved_context(chunks: list[RetrievedChunk]) -> str:
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


class IndustryChainAgent:
    def __init__(self) -> None:
        self.retriever = ChromaRetriever()

    def answer(self, question: str, top_k: int = 5, provider: str | None = None, model: str | None = None) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("question cannot be empty")

        chat_config = resolve_chat_config(provider=provider, model=model)
        client = create_chat_client(provider=chat_config.provider)
        chunks = self.retriever.retrieve(question, top_k=top_k)
        companies = filter_companies(question, load_companies())
        user_prompt = build_report_prompt(
            question=question,
            retrieved_context=format_retrieved_context(chunks),
            company_context=format_company_context(companies),
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
        return {"answer": answer, "sources": sources, "provider": chat_config.provider, "model": chat_config.model}
