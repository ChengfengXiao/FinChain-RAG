from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))

API_URL = "http://localhost:8000/ask"


def post_to_local_api(payload: dict) -> requests.Response:
    session = requests.Session()
    session.trust_env = False
    return session.post(API_URL, json=payload, timeout=120)


st.set_page_config(page_title="FinChain-RAG", layout="wide")

st.title("FinChain-RAG A股产业链研究助手")
st.caption("主题：AI数据中心液冷产业链。仅用于产业研究，不提供买卖建议或股价预测。")

question = st.text_area(
    "输入行业问题",
    value="帮我分析AI数据中心液冷产业链，找出A股核心公司",
    height=120,
)
top_k = st.slider("检索资料数量", min_value=3, max_value=10, value=5)
provider = st.selectbox("生成模型供应商", options=["deepseek", "openai", "minimax"], index=0)
default_models = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-v4-flash",
    "minimax": "MiniMax-M3",
}
model = st.text_input("模型名称", value=default_models[provider])

if st.button("生成分析", type="primary"):
    if not question.strip():
        st.warning("请输入问题。")
    else:
        with st.spinner("正在检索本地知识库并生成研究结果..."):
            try:
                response = post_to_local_api(
                    {"question": question, "top_k": top_k, "provider": provider, "model": model}
                )
                response.raise_for_status()
                payload = response.json()
            except requests.exceptions.ConnectionError:
                st.error("无法连接 FastAPI 服务。请先运行：uvicorn src.api.app:app --reload")
            except requests.exceptions.HTTPError as exc:
                detail = response.json().get("detail", str(exc))
                st.error(f"后端返回错误：{detail}")
            except Exception as exc:
                st.error(f"请求失败：{exc}")
            else:
                st.subheader("分析结果")
                st.caption(f"生成模型：{payload.get('provider')} / {payload.get('model')}")
                st.markdown(payload.get("answer", ""))

                st.subheader("来源")
                sources = payload.get("sources", [])
                if sources:
                    st.dataframe(sources, use_container_width=True)
                else:
                    st.info("暂无来源信息。")
