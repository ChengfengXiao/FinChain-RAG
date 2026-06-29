from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agents.industry_chain_agent import IndustryChainAgent


app = FastAPI(
    title="FinChain-RAG API",
    description="A-share industry chain research assistant powered by RAG, ChromaDB and configurable LLM providers.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户产业链研究问题")
    top_k: int = Field(default=5, ge=1, le=10, description="检索返回的 chunk 数量")
    provider: str | None = Field(default=None, description="LLM provider: openai, deepseek, or minimax")
    model: str | None = Field(default=None, description="可选模型名；不填则使用 provider 默认模型")
    research_mode: str = Field(
        default="local_rag",
        description="local_rag, a_stock_online, bottleneck_hunter, or serenity",
    )
    online_limit: int = Field(default=2, ge=1, le=6, description="在线模式最多抓取的 A 股标的数量")


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    provider: str | None = None
    model: str | None = None
    research_mode: str | None = None
    run_date: str | None = None
    targets: list[str] = []
    operating_snapshots: list[dict[str, Any]] = []
    graph: dict[str, Any] = {}


@lru_cache(maxsize=1)
def get_agent() -> IndustryChainAgent:
    return IndustryChainAgent()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        result = get_agent().answer(
            question=request.question,
            top_k=request.top_k,
            provider=request.provider,
            model=request.model,
            research_mode=request.research_mode,
            online_limit=request.online_limit,
        )
        return AskResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
