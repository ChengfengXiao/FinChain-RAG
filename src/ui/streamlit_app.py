from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
ASK_URL = f"{API_BASE_URL}/ask"
HEALTH_URL = f"{API_BASE_URL}/health"


def local_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def check_api_health() -> tuple[bool, str]:
    try:
        response = local_session().get(HEALTH_URL, timeout=3)
        response.raise_for_status()
        return True, "FastAPI 已连接"
    except Exception as exc:
        return False, f"FastAPI 未连接：{exc}"


def post_to_local_api(payload: dict) -> requests.Response:
    return local_session().post(ASK_URL, json=payload, timeout=240)


def render_source_summary(sources: list[dict]) -> None:
    if not sources:
        st.info("暂无来源信息。")
        return

    ok_count = sum(1 for item in sources if item.get("status") == "ok")
    error_count = sum(1 for item in sources if item.get("status") == "error")
    c1, c2, c3 = st.columns(3)
    c1.metric("数据源总数", len(sources))
    c2.metric("成功", ok_count)
    c3.metric("失败", error_count)
    st.dataframe(sources, use_container_width=True, hide_index=True)


st.set_page_config(page_title="FinChain-RAG", layout="wide")

st.title("FinChain-RAG A股产业链研究工作台")
st.caption("在线模式自动抓 A 股公开数据并调用 DeepSeek；本地 RAG 模式才需要 Chroma 和 markdown 入库。")

api_ok, api_message = check_api_health()
if api_ok:
    st.success(api_message)
else:
    st.error(api_message)
    st.code("uvicorn src.api.app:app --host 127.0.0.1 --port 8000", language="bash")

mode_labels = {
    "bottleneck_hunter": "快速卡点识别",
    "serenity": "Serenity 深度研究",
    "a_stock_online": "A股在线数据",
    "local_rag": "本地 RAG",
}
mode_help = {
    "bottleneck_hunter": "用在线数据快速找产业链最可能卡住的层级，适合先做初筛。",
    "serenity": "用 Serenity 供应链框架做更完整的层级、证据、反方和验证路径。",
    "a_stock_online": "偏数据摘要，适合查个股行情、估值、公告、新闻、资金流和概念。",
    "local_rag": "使用本地 liquid_cooling_docs 入库后的 Chroma 检索结果生成报告。",
}
example_questions = {
    "bottleneck_hunter": "用 bottleneck hunter 快速看英维克和申菱环境在 AI 液冷产业链里到底卡在哪里",
    "serenity": "用 serenity 深度调研 A 股 AI 液冷产业链，先排产业链层级，再给优先研究名单",
    "a_stock_online": "查一下 002837 英维克的行情、概念、公告和资金流线索",
    "local_rag": "请基于本地资料生成一份 AI 数据中心液冷产业链初步研究报告",
}

with st.sidebar:
    st.header("运行设置")
    research_mode = st.radio(
        "研究模式",
        options=list(mode_labels.keys()),
        format_func=lambda key: mode_labels[key],
        index=0,
    )
    st.caption(mode_help[research_mode])

    provider = st.selectbox("模型供应商", options=["deepseek", "openai", "minimax"], index=0)
    default_models = {
        "openai": "gpt-4o-mini",
        "deepseek": "deepseek-v4-flash",
        "minimax": "MiniMax-M3",
    }
    model = st.text_input("模型名称", value=default_models[provider], key=f"model_{provider}")

    if research_mode == "local_rag":
        top_k = st.slider("检索资料数量", min_value=3, max_value=10, value=5)
        online_limit = 1
        st.info("本地 RAG 需要先运行 `python src/ingestion/ingest.py`。")
    else:
        top_k = 5
        online_limit = st.slider("在线抓取标的数", min_value=1, max_value=6, value=2)
        st.caption("建议先用 1-2 个标的。东财接口偶发风控，抓太多会明显变慢。")

question = st.text_area(
    "研究问题",
    value=example_questions[research_mode],
    height=140,
)

left, right = st.columns([1, 3])
with left:
    run_clicked = st.button("生成分析", type="primary", disabled=not api_ok, use_container_width=True)
with right:
    if research_mode == "local_rag":
        st.caption("预计耗时取决于本地向量库和模型响应。")
    else:
        st.caption("预计耗时：单标的约 40-90 秒；多标的会更慢。若东财接口失败，报告仍会基于成功的数据源生成。")

if run_clicked:
    if not question.strip():
        st.warning("请输入问题。")
    else:
        payload = {
            "question": question,
            "top_k": top_k,
            "provider": provider,
            "model": model,
            "research_mode": research_mode,
            "online_limit": online_limit,
        }
        with st.spinner("正在抓取数据并调用模型，请等待..."):
            try:
                response = post_to_local_api(payload)
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.ConnectionError:
                st.error("无法连接 FastAPI 服务。请先启动后端。")
                st.code("uvicorn src.api.app:app --host 127.0.0.1 --port 8000", language="bash")
            except requests.exceptions.Timeout:
                st.error("请求超时。建议把在线抓取标的数降到 1，或稍后重试东财接口。")
            except requests.exceptions.HTTPError as exc:
                try:
                    detail = response.json().get("detail", str(exc))
                except Exception:
                    detail = str(exc)
                st.error(f"后端返回错误：{detail}")
            except Exception as exc:
                st.error(f"请求失败：{exc}")
            else:
                st.subheader("分析结果")
                st.caption(
                    f"运行日期：{result.get('run_date')} | 研究模式：{result.get('research_mode')} | 生成模型：{result.get('provider')} / {result.get('model')}"
                )
                st.markdown(result.get("answer", ""))

                st.subheader("数据源状态")
                render_source_summary(result.get("sources", []))
