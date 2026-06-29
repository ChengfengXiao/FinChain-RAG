from __future__ import annotations

import os
from html import escape
from typing import Any

import pandas as pd
import requests
import streamlit as st


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
        "research_mode": "company_quality",
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


def fmt_ratio(value: Any) -> str:
    if value in {"", None}:
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def metric_card(label: str, value: str, sub: str = "") -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{escape(value)}</div>'
        f'<div class="metric-sub">{escape(sub)}</div>'
        "</div>"
    )


def score_class(score: int) -> str:
    if score >= 85:
        return "score-strong"
    if score >= 70:
        return "score-good"
    if score >= 55:
        return "score-mid"
    return "score-risk"


def render_score(score_data: dict[str, Any]) -> None:
    score = int(score_data.get("score") or 0)
    label = score_data.get("label") or "数据不足"
    details = score_data.get("details") or []
    st.markdown(
        (
            f'<div class="score-panel {score_class(score)}">'
            '<div><div class="score-label">质量评分</div>'
            f'<div class="score-main">{score}<span>/100</span></div></div>'
            f'<div><div class="score-rank">{escape(label)}</div>'
            f'<div class="score-notes">{escape("；".join(details[:3]))}</div></div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_key_metrics(snapshot: dict[str, Any], operating: dict[str, Any]) -> None:
    latest = snapshot.get("latest_period") or {}
    metrics = operating.get("metrics", {})
    cards = [
        metric_card("PE(TTM)", fmt_number(metrics.get("pe_ttm")), "腾讯行情"),
        metric_card("PB", fmt_number(metrics.get("pb")), "腾讯行情"),
        metric_card("营收", fmt_number(latest.get("revenue")), latest.get("report_date", "")),
        metric_card("扣非净利润", fmt_number(latest.get("deduct_net_profit")), latest.get("report_date", "")),
        metric_card("经营现金流/净利润", fmt_ratio(latest.get("ocf_net_profit_ratio")), "现金流覆盖"),
        metric_card("自由现金流", fmt_number(latest.get("free_cashflow")), "OCF - CAPEX"),
        metric_card("资产负债率", fmt_ratio(latest.get("debt_asset_ratio")), "负债安全"),
        metric_card("现金短债比", fmt_number(latest.get("cash_short_debt_ratio")), "短债覆盖"),
    ]
    st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def finance_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = {
        "report_date": "报告期",
        "revenue": "营收",
        "net_profit": "归母净利润",
        "deduct_net_profit": "扣非净利润",
        "operating_cashflow": "经营现金流",
        "free_cashflow": "自由现金流",
        "debt_asset_ratio": "资产负债率",
        "cash_short_debt_ratio": "现金短债比",
        "receivable_revenue_ratio": "应收/营收",
        "inventory_revenue_ratio": "存货/营收",
    }
    data = []
    for row in rows:
        item = {}
        for key, label in columns.items():
            value = row.get(key)
            if key in {"cash_short_debt_ratio"}:
                item[label] = fmt_number(value)
            elif key.endswith("_ratio") or key == "debt_asset_ratio":
                item[label] = fmt_ratio(value)
            elif key == "report_date":
                item[label] = value
            else:
                item[label] = fmt_number(value)
        data.append(item)
    return pd.DataFrame(data)


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.info("暂无数据源。")
        return
    ok = sum(1 for item in sources if item.get("status") == "ok")
    failed = len(sources) - ok
    st.caption(f"成功 {ok} 个，失败 {failed} 个。这里用于核对 AI 结论是否有来源。")
    st.json(sources, expanded=False)


st.set_page_config(page_title="真实利润与现金流企业分析", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; max-width: 1360px;}
    .hero {border:1px solid #e5e7eb; border-radius:8px; padding:22px 26px; background:#ffffff; margin-bottom:18px;}
    .title {font-size:32px; font-weight:760; color:#0f172a; line-height:1.2;}
    .subtitle {color:#475569; margin-top:8px; font-size:15px; max-width:980px;}
    .metric-grid {display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:12px; margin: 16px 0;}
    .metric-card {border:1px solid #e5e7eb; border-radius:8px; padding:14px 16px; background:#fff;}
    .metric-label {font-size:12px; color:#64748b; margin-bottom:8px;}
    .metric-value {font-size:21px; color:#0f172a; font-weight:720; line-height:1.2;}
    .metric-sub {font-size:12px; color:#64748b; margin-top:6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
    .score-panel {display:flex; gap:24px; align-items:center; border-radius:8px; padding:18px 20px; margin:16px 0; border:1px solid #e5e7eb;}
    .score-label {font-size:12px; color:#64748b;}
    .score-main {font-size:44px; font-weight:800; line-height:1; color:#0f172a;}
    .score-main span {font-size:18px; color:#64748b; margin-left:4px;}
    .score-rank {font-size:22px; font-weight:760; color:#0f172a;}
    .score-notes {font-size:13px; color:#475569; margin-top:8px; line-height:1.5;}
    .score-strong {background:#ecfdf5; border-color:#bbf7d0;}
    .score-good {background:#eff6ff; border-color:#bfdbfe;}
    .score-mid {background:#fffbeb; border-color:#fde68a;}
    .score-risk {background:#fef2f2; border-color:#fecaca;}
    .section-title {font-size:20px; font-weight:720; margin:20px 0 8px; color:#0f172a;}
    div[data-testid="stButton"] button {border-radius:8px; height:42px;}
    @media (max-width: 900px) {.metric-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}.score-panel {display:block;}}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
      <div class="title">真实利润与现金流企业分析</div>
      <div class="subtitle">输入 A 股公司名或代码，按最近5年 + 最近4季度框架分析商业模式、行业地位、护城河、真实利润、真实现金流、增长质量、负债风险、财务异常和估值匹配。</div>
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
    with st.spinner("正在抓取最近5年和最近4季度财务数据..."):
        try:
            result = ask_api(query, provider, model)
        except requests.exceptions.HTTPError as exc:
            st.error(f"后端返回错误：{exc.response.text if exc.response is not None else exc}")
            st.stop()
        except Exception as exc:
            st.error(f"请求失败：{exc}")
            st.stop()

    snapshots = result.get("operating_snapshots", [])
    operating = snapshots[0] if snapshots else {}
    financial_list = result.get("financial_quality", [])
    financial = financial_list[0] if financial_list else {}
    score_data = result.get("quality_score", {})
    st.caption(f"运行日期：{result.get('run_date')} | 标的：{', '.join(result.get('targets', [])) or '-'} | 模型：{result.get('provider')} / {result.get('model')}")

    name = operating.get("name") or query
    code = operating.get("code") or (result.get("targets") or [""])[0]
    st.markdown(f'<div class="section-title">{escape(str(name))} · {escape(str(code))}</div>', unsafe_allow_html=True)

    if financial:
        render_score(score_data)
        render_key_metrics(financial, operating)

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<div class="section-title">最近5年年度财务质量</div>', unsafe_allow_html=True)
            st.dataframe(finance_table(financial.get("annual_periods", [])), hide_index=True, use_container_width=True)
        with right:
            st.markdown('<div class="section-title">最近4个季度财务质量</div>', unsafe_allow_html=True)
            st.dataframe(finance_table(financial.get("recent_quarters", [])), hide_index=True, use_container_width=True)
    else:
        st.warning("未能抓取财务质量数据。请确认输入的是 A 股公司名或 6 位代码。")

    st.markdown('<div class="section-title">AI 框架分析报告</div>', unsafe_allow_html=True)
    st.markdown(result.get("answer", ""))

    with st.expander("主营范围和数据源审计"):
        if operating.get("business_scope"):
            st.write(operating.get("business_scope"))
        render_sources(result.get("sources", []))
