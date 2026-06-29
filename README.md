# FinChain-RAG: A股公司运营关系图谱

FinChain-RAG 现在收敛为一个更简单的公司研究工具：输入 A 股公司名或 6 位代码，系统抓取公开数据，生成公司运营情况分析和一层关系图谱。

项目不做荐股，不预测股价，不给买卖建议。所有收入、成本、行情、公告、新闻和资金流描述都必须基于抓取到的公开数据；数据不足时明确说明缺口。

## 核心功能

- 公司搜索：支持公司名或 6 位 A 股代码，例如 `英维克`、`002837`、`宁德时代`。
- 运营情况：展示行情、估值、行业、主营范围、主营构成、收入比例、成本比例和毛利率。
- 一层关系图谱：只保留公司、收入来源、成本去向、ToB/ToC、对应公司。
- 对应公司：来自 `companies.jsonl` 的结构化产业链映射，只作为一层上下游/同主题线索，不等同真实客户或供应商。
- 数据源审计：展示腾讯行情、东财基本面、东财F10经营分析、东财资金流、东财新闻、巨潮公告等接口状态。

## 技术架构

```text
Streamlit UI
    |
    v
FastAPI /ask
    |
    v
AShareDataClient
    |-- 腾讯行情/估值
    |-- 东方财富个股基本面
    |-- 东方财富F10经营分析/主营构成
    |-- 东方财富概念/资金流/新闻
    |-- 巨潮公告
    |
    v
IndustryChainAgent
    |-- 构造运营快照
    |-- 构造一层收入/成本关系图谱
    |-- 调用 DeepSeek/OpenAI/MiniMax 生成分析
```

## 安装

```bash
cd finchain-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

要求 Python 3.10+。

## 环境变量

复制配置：

```bash
cp .env.example .env
```

最小配置：

```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_CHAT_MODEL=deepseek-v4-flash
```

不要把真实 API Key 提交到 GitHub。

## 本地启动

终端 1：启动后端。

```bash
cd /Users/chengfengxiao/Documents/向量数据库/finchain-rag
source .venv/bin/activate
uvicorn src.api.app:app --host 127.0.0.1 --port 8002
```

终端 2：启动前端。

```bash
cd /Users/chengfengxiao/Documents/向量数据库/finchain-rag
source .venv/bin/activate
API_BASE_URL=http://127.0.0.1:8002 streamlit run src/ui/streamlit_app.py
```

打开：

```text
http://localhost:8501
```

注意：`8002` 是 API 后端，`8501` 是网页前端。

## API 示例

```http
POST /ask
Content-Type: application/json

{
  "question": "英维克",
  "provider": "deepseek",
  "model": "deepseek-v4-flash"
}
```

返回字段：

- `answer`：AI 运营分析。
- `targets`：识别到的 A 股代码。
- `operating_snapshots`：公司运营快照。
- `graph`：一层关系图谱节点和边。
- `sources`：数据源审计。

## 图谱规则

默认只分析 1 层关系：

- 公司 -> 收入来源：来自东财F10主营构成的主营收入、收入比例、毛利率。
- 成本去向 -> 公司：来自东财F10主营构成的主营成本和成本比例。
- 公司 -> ToB/ToC：基于主营范围、主营构成、行业/概念文本做保守判断。
- 公司 <-> 对应公司：来自结构化产业链映射，仅表示上下游/同主题线索。

如果公开数据没有披露具体客户或供应商，系统会明确说明“未披露”，不会把产业链线索说成真实客户或供应商。

## 示例输入

- `英维克`
- `002837`
- `宁德时代`
- `300750`

## 部署到 Render

项目已包含 `render.yaml`。在 Render 创建 Blueprint，选择 GitHub 仓库后填写：

```bash
DEEPSEEK_API_KEY=你的 DeepSeek Key
```

前端服务的环境变量：

```bash
API_BASE_URL=https://你的-api服务地址
```

重新部署后访问 Streamlit 服务 URL 即可。

## 免责声明

本项目仅用于公开数据整理和公司运营研究辅助，不构成任何投资建议。公开接口可能存在延迟、缺失或临时失败，使用时应结合公司公告、年报、招股书和交易所披露文件复核。
