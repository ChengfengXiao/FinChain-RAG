# FinChain-RAG: A-share Industry Chain Research Assistant

FinChain-RAG 是一个基于 RAG、向量数据库和 LLM 的 A股产业链研究助手。第一版聚焦“AI数据中心液冷产业链”，支持用户输入产业研究问题后，基于本地知识库检索资料，并输出产业链上下游拆解、A股公司映射、龙头公司排序、研究逻辑和引用来源。

本项目定位是产业研究助手，不做股票价格预测，不提供买入、卖出或持有建议。

## 功能特点

- 本地知识库：内置 AI数据中心液冷产业链 markdown 资料。
- 数据入库 pipeline：清洗文档、切分 chunks、调用本地 embedding 模型、写入 ChromaDB。
- RAG 检索：用户问题向量化后，从 ChromaDB 检索 top-k 相关片段。
- 结构化公司数据：使用 `companies.jsonl` 管理 A股公司、产业链环节、相关逻辑、评分和风险提示。
- 产业链 Agent：结合检索资料和公司结构化数据，生成初步投研报告。
- FastAPI 后端：提供 `/ask` 问答接口。
- Streamlit 前端：提供可视化问答页面。
- 明确引用来源：回答返回 `sources`，便于追溯本地知识库片段。

## 技术架构图

```text
User Question
    |
    v
Streamlit UI  ----HTTP---->  FastAPI /ask
                                |
                                v
                       IndustryChainAgent
                         |            |
                         |            v
                         |     companies.jsonl
                         v
                    ChromaRetriever
                         |
                         v
                    ChromaDB Vector Store
                         ^
                         |
              Ingestion Pipeline
                         ^
                         |
        data/raw/liquid_cooling_docs/*.md

Model APIs:
- local sentence-transformers model for embeddings by default
- DeepSeek by default for report generation
- OpenAI / MiniMax can be selected as optional Chat providers
```

## RAG 与向量数据库如何体现

本项目的 RAG 流程不是只把资料塞进 prompt，而是完整走了“入库、向量检索、上下文增强、LLM生成”：

1. 文档入库：`src/ingestion/ingest.py` 读取 `data/raw/liquid_cooling_docs/` 下的 markdown 文档。
2. 文本清洗与切分：脚本把行业资料清洗后切成 chunks，默认 `chunk_size=500`、`overlap=80`。
3. 向量化：每个 chunk 使用本地 sentence-transformers 模型生成 embedding，默认 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。
4. 向量数据库：chunk、metadata 和 embedding 写入本地 `chroma_db/`，使用 ChromaDB 持久化保存。
5. 查询检索：`src/retriever/retriever.py` 将用户问题向量化，从 ChromaDB 检索 top-k 相关 chunks。
6. 上下文增强：`src/agents/industry_chain_agent.py` 把检索 chunks 和 `companies.jsonl` 公司结构化数据一起放入 prompt。
7. LLM生成：默认使用 DeepSeek，模型只基于检索资料和公司数据输出研究报告，并返回 `sources`。

关键文件：

- 向量入库：`src/ingestion/ingest.py`
- 向量检索：`src/retriever/retriever.py`
- RAG Agent：`src/agents/industry_chain_agent.py`
- Prompt 模板：`src/prompts/report_prompt.py`
- 向量库目录：`chroma_db/`

## 安装方法

```bash
cd finchain-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

要求 Python 3.10+。

## .env 配置方法

复制示例配置：

```bash
cp .env.example .env
```

编辑 `.env`：

```bash
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_CHAT_MODEL=deepseek-v4-flash

CHROMA_DB_DIR=chroma_db
CHROMA_COLLECTION_NAME=liquid_cooling_industry
```

不要把真实 API Key 提交到 GitHub。

### 多模型配置

当前版本支持切换 3 个 Chat 模型供应商：

- `openai`
- `deepseek`
- `minimax`

注意：embedding 入库和检索默认使用本地 sentence-transformers 模型，不需要 OpenAI Key。第一次运行入库时会下载模型，可能需要等待几分钟。

OpenAI：

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_CHAT_MODEL=gpt-4o-mini
```

DeepSeek：

```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_CHAT_MODEL=deepseek-v4-flash
```

MiniMax：

```bash
LLM_PROVIDER=minimax
MINIMAX_API_KEY=your-minimax-api-key
MINIMAX_BASE_URL=https://api.minimax.io/v1
MINIMAX_CHAT_MODEL=MiniMax-M3
```

也可以在 Streamlit 页面里直接选择 `openai`、`deepseek` 或 `minimax`，并手动填写模型名。FastAPI `/ask` 也支持传入：

```json
{
  "question": "请生成一份AI液冷产业链初步研究报告",
  "provider": "deepseek",
  "model": "deepseek-v4-flash",
  "top_k": 5
}
```

## 数据入库方法

项目已内置 5 篇 AI数据中心液冷产业链资料：

- `data/raw/liquid_cooling_docs/01_why_liquid_cooling.md`
- `data/raw/liquid_cooling_docs/02_upstream_components.md`
- `data/raw/liquid_cooling_docs/03_midstream_systems.md`
- `data/raw/liquid_cooling_docs/04_downstream_applications.md`
- `data/raw/liquid_cooling_docs/05_ashare_companies.md`

执行入库：

```bash
python src/ingestion/ingest.py
```

默认参数：

- `chunk_size=500`
- `overlap=80`
- metadata 包含 `source`、`chunk_index`、`theme`

默认使用本地 embedding，不需要配置 `OPENAI_API_KEY`。如果你把 `EMBEDDING_PROVIDER` 改为 `openai`，才需要配置 OpenAI Key。

## 启动 FastAPI 方法

```bash
uvicorn src.api.app:app --reload
```

接口：

```http
POST /ask
Content-Type: application/json

{
  "question": "帮我分析AI数据中心液冷产业链，找出A股核心公司"
}
```

响应：

```json
{
  "answer": "...",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "sources": [
    {
      "source": "03_midstream_systems.md",
      "chunk_index": 0,
      "theme": "AI数据中心液冷",
      "distance": 0.25
    }
  ]
}
```

## 启动 Streamlit 方法

先启动 FastAPI，再运行：

```bash
streamlit run src/ui/streamlit_app.py
```

页面包含：

- 标题：FinChain-RAG A股产业链研究助手
- 行业问题输入框
- 生成分析按钮
- 分析结果展示区
- 来源展示区

## 示例问题

- AI数据中心为什么需要液冷？
- 液冷产业链上游有哪些环节？
- AI液冷相关的A股公司有哪些？
- 帮我按照上游、中游、下游拆解液冷产业链
- 哪些环节最可能卡住AI数据中心液冷发展？
- 请生成一份AI液冷产业链初步研究报告

## 示例输出

更完整的示例见 `examples/liquid_cooling_report_example.md`。

```markdown
## 1. 核心结论

AI服务器功率密度提升推动液冷从可选方案变成高密度算力部署的重要基础设施。产业机会集中在上游可靠零部件、中游CDU/冷板/液冷机柜，以及下游智算中心建设。

## 2. 产业链拆解

### 上游
冷却液、泵、阀、管路、快接头、金属材料和密封件。

### 中游
CDU、冷板、manifold、液冷机柜和温控系统。

### 下游
AI服务器、IDC数据中心、云厂商和算力中心。
```

## 项目亮点

- 端到端 RAG 流程完整：从文档入库、向量检索到 LLM 生成均可本地运行。
- 结合非结构化资料和结构化公司 JSONL，展示产业研究中常见的数据融合方式。
- Prompt 单独管理，明确约束“不编造、不荐股、不预测股价”。
- 回答返回引用来源，便于追溯知识库片段。
- 后端和前端分离，适合作为简历项目继续扩展。

## 后续可扩展方向

- 增加更多产业主题，例如光模块、HBM、先进封装、机器人和电力设备。
- 引入更严格的文档解析和清洗流程，支持 PDF、研报、公告和新闻。
- 增加 rerank 模型，提高复杂问题下的检索质量。
- 将公司数据扩展为财务指标、业务占比、客户验证阶段和订单进展。
- 引入 Qdrant 或 Milvus，支持更大规模向量检索。
- 增加 FastAPI 鉴权、任务队列、缓存和日志系统。
- 增加前端图表，展示产业链环节、公司评分和来源覆盖情况。

## 免责声明

本项目仅用于技术展示和产业研究辅助，不构成任何投资建议。A股公司映射和评分来自本地模拟资料与结构化示例数据，不代表真实投资价值排序。
