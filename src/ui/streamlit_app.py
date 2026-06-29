from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


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


def ask_api(company_query: str, provider: str, model: str) -> dict[str, Any]:
    payload = {
        "question": company_query,
        "provider": provider,
        "model": model,
        "research_mode": "company_ops",
        "online_limit": 1,
    }
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
        st.info("暂无关系图谱。通常是未识别到 A 股公司或公开接口失败。")
        return

    colors = {
        "company": "#1d4ed8",
        "revenue": "#059669",
        "cost": "#dc2626",
        "model": "#7c3aed",
        "related": "#64748b",
    }
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#111827",
        directed=True,
        cdn_resources="in_line",
    )
    net.barnes_hut(gravity=-22000, central_gravity=0.22, spring_length=150, spring_strength=0.04, damping=0.18)
    for node in nodes:
        group = node.get("group", "related")
        net.add_node(
            node["id"],
            label=node.get("label", node["id"]),
            title=node.get("title", ""),
            color=colors.get(group, "#64748b"),
            value=node.get("value", 14),
        )
    for edge in edges:
        net.add_edge(edge["from"], edge["to"], label=edge.get("label", ""), title=edge.get("title", ""), color="#94a3b8")

    with NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tmp:
        net.save_graph(tmp.name)
        html = Path(tmp.name).read_text(encoding="utf-8")
    components.html(html, height=630, scrolling=False)


def render_composition(title: str, rows: list[dict[str, Any]], ratio_key: str, amount_key: str) -> None:
    st.markdown(f"#### {title}")
    if not rows:
        st.caption("公开数据未返回该构成。")
        return
    table = []
    for row in rows:
        table.append(
            {
                "项目": row.get("item_name"),
                "报告期": row.get("report_date"),
                "金额": fmt_number(row.get(amount_key)),
                "占比": f"{(row.get(ratio_key) or 0) * 100:.2f}%",
                "毛利率": f"{(row.get('gross_margin') or 0) * 100:.2f}%",
            }
        )
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)


def render_snapshot(snapshot: dict[str, Any]) -> None:
    metrics = snapshot.get("metrics", {})
    business_model = snapshot.get("business_model", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("最新价", fmt_number(metrics.get("price")))
    c2.metric("涨跌幅", fmt_number(metrics.get("change_pct"), "%"))
    c3.metric("PE(TTM)", fmt_number(metrics.get("pe_ttm")))
    c4.metric("PB", fmt_number(metrics.get("pb")))
    c5.metric("ToB / ToC", business_model.get("model", "不确定"))

    st.caption(f"行业：{metrics.get('industry') or '-'} | 主营构成报告期：{snapshot.get('composition_date') or '-'}")
    if snapshot.get("business_scope"):
        with st.expander("主营范围"):
            st.write(snapshot.get("business_scope"))

    left, right = st.columns(2)
    with left:
        render_composition("收入来自哪里", snapshot.get("revenue_mix", []), "revenue_ratio", "revenue_yuan")
    with right:
        render_composition("成本/支出主要去向", snapshot.get("cost_mix", []), "cost_ratio", "cost_yuan")


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.info("暂无数据源。")
        return
    rows = pd.DataFrame(sources)
    st.dataframe(rows, use_container_width=True, hide_index=True)


st.set_page_config(page_title="FinChain-RAG 公司运营图谱", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; max-width: 1440px;}
    div[data-testid="stMetric"] {background: #fff; border: 1px solid #e5e7eb; padding: 12px 14px; border-radius: 8px;}
    section[data-testid="stSidebar"] {background: #f8fafc;}
    .title {font-size: 30px; font-weight: 760; color: #111827;}
    .subtitle {color: #475569; margin-bottom: 18px;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="title">FinChain-RAG 公司运营关系图谱</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">只分析一个公司：运营情况、收入来源、成本去向、ToB/ToC，以及一层上下游/对应公司线索。</div>',
    unsafe_allow_html=True,
)

api_ok, api_message = check_api_health()
st.success(api_message) if api_ok else st.error(api_message)

with st.sidebar:
    st.header("设置")
    provider = st.selectbox("模型供应商", ["deepseek", "openai", "minimax"], index=0)
    default_models = {"deepseek": "deepseek-v4-flash", "openai": "gpt-4o-mini", "minimax": "MiniMax-M3"}
    model = st.text_input("模型名称", value=default_models[provider])
    st.caption("关系图谱默认只展示 1 层：收入来源、成本去向、ToB/ToC、对应公司。")

examples = ["英维克", "002837", "宁德时代", "300750"]
query = st.text_input("输入公司名或 A 股代码", value="英维克", placeholder="例如：英维克、002837、宁德时代")
cols = st.columns(len(examples))
for idx, example in enumerate(examples):
    if cols[idx].button(example, use_container_width=True):
        query = example

run_clicked = st.button("生成公司运营图谱", type="primary", disabled=not api_ok, use_container_width=True)

if run_clicked:
    with st.spinner("正在抓取公开数据并生成图谱..."):
        try:
            result = ask_api(query, provider, model)
        except requests.exceptions.HTTPError as exc:
            st.error(f"后端返回错误：{exc.response.text if exc.response is not None else exc}")
            st.stop()
        except Exception as exc:
            st.error(f"请求失败：{exc}")
            st.stop()

    snapshots = result.get("operating_snapshots", [])
    graph = result.get("graph", {})
    st.caption(f"运行日期：{result.get('run_date')} | 标的：{', '.join(result.get('targets', [])) or '-'} | 模型：{result.get('provider')} / {result.get('model')}")

    if snapshots:
        snapshot = snapshots[0]
        st.subheader(f"{snapshot.get('name') or query} `{snapshot.get('code', '')}`")
        render_snapshot(snapshot)
    else:
        st.warning("未能生成运营快照。请确认输入的是 A 股公司名或 6 位代码。")

    tab_graph, tab_report, tab_sources = st.tabs(["一层关系图谱", "AI 运营分析", "数据源"])
    with tab_graph:
        render_graph(graph)
    with tab_report:
        st.markdown(result.get("answer", ""))
    with tab_sources:
        render_sources(result.get("sources", []))
