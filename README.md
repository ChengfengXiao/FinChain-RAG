# FinChain-RAG: A股真实利润与现金流企业分析

FinChain-RAG 现在定位为一个 A 股公司质量分析工具：输入公司名或 6 位股票代码，系统抓取公开数据，并按“真实利润与现金流企业分析框架”输出公司研究报告。

项目不做荐股，不预测股价，不给买卖建议。分析重点是真实利润、真实现金流、扣非净利润、自由现金流、增长质量、负债安全和财务异常，而不是概念故事。

## 核心功能

- 公司搜索：支持公司名或 6 位 A 股代码，例如 `英维克`、`002837`、`宁德时代`。
- 最近5年年度财务：营收、归母净利润、扣非净利润、经营现金流、自由现金流、负债率等。
- 最近4季度财务：跟踪最新利润、现金流、应收、存货和负债变化。
- 质量评分：输出 100 分评分，并分类为顶级公司、优秀公司、普通公司、高风险公司、应回避公司。
- AI 框架报告：覆盖商业模式、行业地位、护城河、真实利润、真实现金流、增长质量、负债风险、财务异常和估值判断。
- 数据源审计：展示腾讯行情、东财基本面、东财F10经营分析、东财三表摘要、新浪三表、新闻和公告接口状态。

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
    |-- 东方财富财务摘要：利润表/现金流量表/资产负债表
    |-- 新浪财报三表补充字段
    |-- 东方财富新闻/资金流 + 巨潮公告
    |
    v
IndustryChainAgent
    |-- 合并最近5年年度财务
    |-- 合并最近4季度财务
    |-- 计算真实利润/现金流/负债/异常指标
    |-- 调用 DeepSeek/OpenAI/MiniMax 生成框架报告
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
  "model": "deepseek-v4-flash",
  "research_mode": "company_quality"
}
```

返回字段：

- `answer`：AI 框架分析报告。
- `targets`：识别到的 A 股代码。
- `operating_snapshots`：主营业务、行情、估值、行业和公告新闻摘要。
- `financial_quality`：最近5年 + 最近4季度财务质量数据。
- `quality_score`：预评分、分类和关键评分依据。
- `sources`：数据源审计。

## 分析框架

输出报告固定覆盖：

1. 商业模式：公司靠什么赚钱，收入是否可持续。
2. 行业地位：是否可能属于行业龙头，市占率和竞争格局是否有数据支撑。
3. 护城河：品牌、成本、技术、渠道、规模、资源、客户粘性。
4. 真实利润：净利润和扣非净利润是否匹配，利润是否来自主营业务。
5. 真实现金流：经营现金流是否覆盖净利润，自由现金流是否为正。
6. 增长质量：营收、扣非净利润、经营现金流是否同步增长。
7. 负债风险：资产负债率、有息负债、现金短债比、利息保障倍数。
8. 财务异常：应收账款、存货、商誉、资本开支是否异常。
9. 估值判断：PE/PB/市值是否匹配公司质量和增长。
10. 最终结论：100 分评分和公司质量分类。

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

本项目仅用于公开数据整理和公司研究辅助，不构成任何投资建议。公开接口可能存在延迟、缺失或临时失败，使用时应结合公司公告、年报、招股书和交易所披露文件复核。
