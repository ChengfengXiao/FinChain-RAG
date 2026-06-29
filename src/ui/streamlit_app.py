from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

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


def pct(value: Any) -> float:
    try:
        return max(0.0, min(float(value or 0) * 100, 100.0))
    except (TypeError, ValueError):
        return 0.0


def card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>
    """


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
    st.markdown(f"### {title}")
    if not rows:
        st.caption("公开数据未返回该构成。")
        return
    blocks = []
    for row in rows:
        ratio = pct(row.get(ratio_key))
        blocks.append(
            f"""
            <div class="mix-row">
              <div class="mix-top">
                <span class="mix-name">{row.get("item_name") or "-"}</span>
                <span class="mix-ratio">{ratio:.1f}%</span>
              </div>
              <div class="bar"><div class="bar-fill" style="width:{ratio:.1f}%"></div></div>
              <div class="mix-meta">
                <span>{fmt_number(row.get(amount_key))}</span>
                <span>毛利率 {(row.get("gross_margin") or 0) * 100:.1f}%</span>
              </div>
            </div>
            """
        )
    st.markdown("".join(blocks), unsafe_allow_html=True)


def render_snapshot(snapshot: dict[str, Any]) -> None:
    metrics = snapshot.get("metrics", {})
    business_model = snapshot.get("business_model", {})
    st.markdown(
        f"""
        <div class="metric-grid">
          {card("最新价", fmt_number(metrics.get("price")), "腾讯行情")}
          {card("涨跌幅", fmt_number(metrics.get("change_pct"), "%"), "腾讯行情")}
          {card("PE(TTM)", fmt_number(metrics.get("pe_ttm")), "腾讯行情")}
          {card("PB", fmt_number(metrics.get("pb")), "腾讯行情")}
          {card("ToB / ToC", business_model.get("model", "不确定"), business_model.get("evidence", ""))}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="info-strip">
          <span>行业：{metrics.get("industry") or "-"}</span>
          <span>主营构成报告期：{snapshot.get("composition_date") or "-"}</span>
          <span>判断置信度：{business_model.get("confidence", "-")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if snapshot.get("business_scope"):
        with st.expander("主营范围，来自东财F10"):
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
    ok = sum(1 for item in sources if item.get("status") == "ok")
    failed = len(sources) - ok
    st.caption(f"成功 {ok} 个，失败 {failed} 个。这里用于核对 AI 结论是否有来源。")
    st.json(sources, expanded=False)


st.set_page_config(page_title="FinChain-RAG 公司运营图谱", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; max-width: 1380px;}
    section[data-testid="stSidebar"] {background: #f8fafc;}
    .hero {
      border: 1px solid #e5e7eb; border-radius: 8px; padding: 22px 26px;
      background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
      margin-bottom: 18px;
    }
    .title {font-size: 34px; font-weight: 760; color: #0f172a; line-height: 1.15;}
    .subtitle {color: #475569; margin-top: 8px; font-size: 15px;}
    .metric-grid {display:grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap:12px; margin: 16px 0;}
    .metric-card {border:1px solid #e5e7eb; border-radius:8px; padding:14px 16px; background:#fff;}
    .metric-label {font-size:12px; color:#64748b; margin-bottom:8px;}
    .metric-value {font-size:22px; color:#0f172a; font-weight:720; line-height:1.2;}
    .metric-sub {font-size:12px; color:#64748b; margin-top:6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
    .info-strip {display:flex; gap:20px; border:1px solid #e5e7eb; border-radius:8px; padding:10px 14px; color:#475569; background:#f8fafc; margin-bottom:16px;}
    .mix-row {border:1px solid #e5e7eb; border-radius:8px; padding:12px 14px; background:#fff; margin-bottom:10px;}
    .mix-top {display:flex; justify-content:space-between; gap:12px; align-items:center;}
    .mix-name {font-weight:650; color:#0f172a;}
    .mix-ratio {font-weight:700; color:#0f766e;}
    .bar {height:8px; background:#e2e8f0; border-radius:999px; overflow:hidden; margin:10px 0 8px;}
    .bar-fill {height:100%; background:#0f766e; border-radius:999px;}
    .mix-meta {display:flex; justify-content:space-between; color:#64748b; font-size:12px;}
    .section-title {font-size:20px; font-weight:720; margin:18px 0 8px; color:#0f172a;}
    div[data-testid="stButton"] button {border-radius:8px; height:42px;}
    @media (max-width: 900px) {.metric-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}.info-strip {display:block;}}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
      <div class="title">公司运营关系图谱</div>
      <div class="subtitle">输入 A 股公司名或代码，查看收入来自哪里、成本花在哪里、业务更偏 ToB 还是 ToC，以及一层对应公司线索。</div>
    </div>
    """,
    unsafe_allow_html=True,
)

api_ok, api_message = check_api_health()
if api_ok:
    st.success(api_message)
else:
    st.error(api_message)

with st.sidebar:
    st.header("模型设置")
    provider = st.selectbox("模型供应商", ["deepseek", "openai", "minimax"], index=0)
    default_models = {"deepseek": "deepseek-v4-flash", "openai": "gpt-4o-mini", "minimax": "MiniMax-M3"}
    model = st.text_input("模型名称", value=default_models[provider])
    st.caption("默认使用 DeepSeek。普通使用不需要改这里。")

examples = ["英维克", "002837", "宁德时代", "300750"]
search_col, button_col = st.columns([5, 1.5], vertical_alignment="bottom")
with search_col:
    query = st.text_input("公司名或 A 股代码", value="英维克", placeholder="例如：英维克、002837、宁德时代", label_visibility="collapsed")
with button_col:
    run_clicked = st.button("生成分析", type="primary", disabled=not api_ok, use_container_width=True)

cols = st.columns(len(examples))
for idx, example in enumerate(examples):
    if cols[idx].button(example, use_container_width=True):
        query = example

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
        st.markdown(f'<div class="section-title">{snapshot.get("name") or query} · {snapshot.get("code", "")}</div>', unsafe_allow_html=True)
        render_snapshot(snapshot)
    else:
        st.warning("未能生成运营快照。请确认输入的是 A 股公司名或 6 位代码。")

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown('<div class="section-title">一层关系图谱</div>', unsafe_allow_html=True)
        render_graph(graph)
    with right:
        st.markdown('<div class="section-title">AI 运营分析</div>', unsafe_allow_html=True)
        st.markdown(result.get("answer", ""))

    with st.expander("数据源审计"):
        render_sources(result.get("sources", []))
