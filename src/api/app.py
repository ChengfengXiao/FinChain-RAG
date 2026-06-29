from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agents.industry_chain_agent import IndustryChainAgent


app = FastAPI(
    title="FinChain-RAG API",
    description="A-share real profit and cash flow quality analysis API.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="公司名或 A 股代码")
    top_k: int = Field(default=5, ge=1, le=10, description="保留字段，当前公司运营模式不使用")
    provider: str | None = Field(default=None, description="LLM provider: openai, deepseek, or minimax")
    model: str | None = Field(default=None, description="可选模型名；不填则使用 provider 默认模型")
    research_mode: str = Field(
        default="company_quality",
        description="固定为 company_quality：真实利润与现金流企业分析框架",
    )
    online_limit: int = Field(default=1, ge=1, le=1, description="当前默认只分析 1 个公司")


class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    provider: str | None = None
    model: str | None = None
    research_mode: str | None = None
    run_date: str | None = None
    targets: list[str] = []
    operating_snapshots: list[dict[str, Any]] = []
    financial_quality: list[dict[str, Any]] = []
    quality_score: dict[str, Any] = {}


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
