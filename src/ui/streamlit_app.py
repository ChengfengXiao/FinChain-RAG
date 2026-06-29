from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

sys.path.append(str(Path(__file__).resolve().parents[2]))

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
ASK_URL = f"{API_BASE_URL}/ask"
HEALTH_URL = f"{API_BASE_URL}/health"


def api_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def check_api_health() -> tuple[bool, str]:
    try:
        response = api_session().get(HEALTH_URL, timeout=3)
        response.raise_for_status()
        return True, "后端已连接"
    except Exception as exc:
        return False, f"后端未连接：{exc}"


def ask_api(payload: dict[str, Any]) -> dict[str, Any]:
    response = api_session().post(ASK_URL, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def fmt_number(value: Any, unit: str = "") -> str:
    if value in {"", None}:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 100000000:
        return f"{number / 100000000:.2f}亿{unit}"
    if abs(number) >= 10000:
        return f"{number / 10000:.2f}万{unit}"
    return f"{number:.2f}{unit}"


def render_graph(graph: dict[str, Any]) -> None:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not nodes:
        st.info("当前模式没有返回关系图谱。请选择在线研究模式，或扩大抓取标的数。")
        return

    colors = {
        "query": "#111827",
        "company": "#2563eb",
        "industry": "#059669",
        "segment": "#7c3aed",
        "concept": "#d97706",
        "evidence": "#64748b",
    }
    net = Network(
        height="620px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#111827",
        directed=True,
        cdn_resources="in_line",
    )
    net.barnes_hut(gravity=-25000, central_gravity=0.2, spring_length=160, spring_strength=0.035, damping=0.18)

    for node in nodes:
        group = node.get("group", "evidence")
        net.add_node(
            node["id"],
            label=node.get("label", node["id"]),
            title=node.get("title", ""),
            color=colors.get(group, "#64748b"),
            value=node.get("value", 16),
        )
    for edge in edges:
        net.add_edge(edge["from"], edge["to"], label=edge.get("label", ""), title=edge.get("title", ""), color="#94a3b8")

    with NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tmp:
        net.save_graph(tmp.name)
        html = Path(tmp.name).read_text(encoding="utf-8")
    components.html(html, height=650, scrolling=False)


def render_snapshots(snapshots: list[dict[str, Any]]) -> None:
    if not snapshots:
        st.info("暂无运营快照。在线数据接口失败或未识别到 A 股标的时会出现这种情况。")
        return

    for item in snapshots:
        metrics = item.get("metrics", {})
        with st.container(border=True):
            st.markdown(f"### {item.get('name') or item.get('code')} `{item.get('code')}`")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("最新价", fmt_number(metrics.get("price")))
            c2.metric("涨跌幅", fmt_number(metrics.get("change_pct"), "%"))
            c3.metric("PE(TTM)", fmt_number(metrics.get("pe_ttm")))
            c4.metric("PB", fmt_number(metrics.get("pb")))

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("总市值", fmt_number(metrics.get("mcap_yuan") or ((metrics.get("mcap_yi") or 0) * 100000000)))
            c6.metric("换手率", fmt_number(metrics.get("turnover_pct"), "%"))
            c7.metric("5日主力净流", fmt_number(metrics.get("main_net_5d_yuan")))
            c8.metric("20日主力净流", fmt_number(metrics.get("main_net_20d_yuan")))

            concepts = item.get("concepts", [])
            if concepts:
                st.caption("概念/板块：" + " / ".join(concepts[:10]))

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**近期公告**")
                announcements = item.get("announcements", [])
                if announcements:
                    for ann in announcements[:3]:
                        st.markdown(f"- {ann.get('date', '')} [{ann.get('title', '')}]({ann.get('url', '')})")
                else:
                    st.caption("未抓取到公告。")
            with col_b:
                st.markdown("**近期新闻**")
                news = item.get("news", [])
                if news:
                    for row in news[:3]:
                        st.markdown(f"- {row.get('time', '')} [{row.get('title', '')}]({row.get('url', '')})")
                else:
                    st.caption("未抓取到新闻。")


def render_source_summary(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.info("暂无来源信息。")
        return
    ok_count = sum(1 for item in sources if item.get("status") == "ok")
    error_count = sum(1 for item in sources if item.get("status") == "error")
    c1, c2, c3 = st.columns(3)
    c1.metric("数据源", len(sources))
    c2.metric("成功", ok_count)
    c3.metric("失败", error_count)
    st.dataframe(pd.DataFrame(sources), use_container_width=True, hide_index=True)


st.set_page_config(page_title="FinChain-RAG", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; max-width: 1480px;}
    div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e5e7eb; padding: 14px 16px; border-radius: 8px;}
    section[data-testid="stSidebar"] {background: #f8fafc;}
    .finchain-hero {border-bottom: 1px solid #e5e7eb; padding-bottom: 16px; margin-bottom: 18px;}
    .finchain-title {font-size: 32px; font-weight: 750; color: #111827; margin-bottom: 4px;}
    .finchain-subtitle {font-size: 15px; color: #475569;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="finchain-hero">
      <div class="finchain-title">FinChain-RAG 产业图谱研究台</div>
      <div class="finchain-subtitle">输入产业或 A 股公司，系统抓取公开数据，生成关系图谱、运营快照和投研初筛结论。</div>
    </div>
    """,
    unsafe_allow_html=True,
)

api_ok, api_message = check_api_health()
if api_ok:
    st.success(api_message)
else:
    st.error(api_message)
    st.code("uvicorn src.api.app:app --host 127.0.0.1 --port 8000", language="bash")

mode_labels = {
    "bottleneck_hunter": "产业卡点图谱",
    "serenity": "深度产业链研究",
    "a_stock_online": "公司运营快照",
    "local_rag": "本地 RAG 报告",
}

with st.sidebar:
    st.header("研究设置")
    research_mode = st.radio("分析模式", list(mode_labels.keys()), format_func=lambda key: mode_labels[key], index=0)
    online_limit = st.slider("抓取标的数", min_value=1, max_value=6, value=3, disabled=research_mode == "local_rag")
    top_k = st.slider("本地检索数量", min_value=3, max_value=10, value=5, disabled=research_mode != "local_rag")

    st.divider()
    provider = st.selectbox("模型供应商", ["deepseek", "openai", "minimax"], index=0)
    default_models = {"deepseek": "deepseek-v4-flash", "openai": "gpt-4o-mini", "minimax": "MiniMax-M3"}
    model = st.text_input("模型名称", value=default_models[provider], key=f"model_{provider}")

    st.divider()
    st.caption("线上部署时只需要配置 API_BASE_URL 和 DEEPSEEK_API_KEY。公开数据接口偶发失败时，系统会在来源审计中标出。")

examples = [
    "液冷产业链有哪些 A 股公司？生成关系图谱并说明谁更接近真实瓶颈",
    "英维克 002837 的运营情况、概念、公告和资金流怎么样？",
    "机器人产业链里哪些 A 股公司值得优先研究？",
    "宁德时代的公司运营情况和产业关系图谱",
]
query = st.text_input("搜索产业或公司", value=examples[0], placeholder="例如：液冷、英维克、002837、机器人产业链")
quick_cols = st.columns(len(examples))
for idx, example in enumerate(examples):
    if quick_cols[idx].button(example[:12] + "...", use_container_width=True):
        query = example

run_clicked = st.button("生成图谱和研究结论", type="primary", disabled=not api_ok, use_container_width=True)

if run_clicked:
    payload = {
        "question": query,
        "top_k": top_k,
        "provider": provider,
        "model": model,
        "research_mode": research_mode,
        "online_limit": online_limit,
    }
    with st.spinner("正在抓取公开数据、构建图谱并调用模型..."):
        try:
            result = ask_api(payload)
        except requests.exceptions.ConnectionError:
            st.error("无法连接 FastAPI 后端。")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("请求超时。建议减少抓取标的数，或稍后重试。")
            st.stop()
        except requests.exceptions.HTTPError as exc:
            st.error(f"后端返回错误：{exc.response.text if exc.response is not None else exc}")
            st.stop()
        except Exception as exc:
            st.error(f"请求失败：{exc}")
            st.stop()

    st.caption(
        f"运行日期：{result.get('run_date')} | 模式：{mode_labels.get(result.get('research_mode'), result.get('research_mode'))} | 模型：{result.get('provider')} / {result.get('model')}"
    )

    targets = result.get("targets", [])
    sources = result.get("sources", [])
    graph = result.get("graph", {})
    snapshots = result.get("operating_snapshots", [])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("识别标的", len(targets))
    c2.metric("图谱节点", len(graph.get("nodes", [])))
    c3.metric("图谱关系", len(graph.get("edges", [])))
    c4.metric("数据源", len(sources))

    tab_graph, tab_ops, tab_report, tab_sources = st.tabs(["关系图谱", "运营情况", "AI研究报告", "数据源审计"])
    with tab_graph:
        render_graph(graph)
    with tab_ops:
        render_snapshots(snapshots)
    with tab_report:
        st.markdown(result.get("answer", ""))
    with tab_sources:
        render_source_summary(sources)
