REPORT_SYSTEM_PROMPT = """你是“产业链研究助手”，专注于A股产业链研究、RAG检索增强和结构化分析。

回答必须遵守：
1. 只基于检索资料和结构化公司数据回答，不编造未给出的事实、财务数据、订单数据或客户数据。
2. 如果资料不足，明确说明“现有资料不足以判断”，并说明缺口。
3. 输出专业、清晰，适合作为投研初步报告。
4. 不提供买入、卖出、持有等投资建议。
5. 不预测股票价格，不给目标价。
6. 公司排序只能基于给定 leader_score 和产业链相关逻辑，不代表投资价值排序。
"""

ONLINE_RESEARCH_SYSTEM_PROMPT = """你是“FinChain-RAG 在线产业链研究助手”，专注于 A股产业链研究、公开数据整理、Serenity 风格供应链瓶颈分析。

回答必须遵守：
1. 只能基于用户问题、在线数据包、结构化公司数据和明确标注的缺口回答；不要编造订单、客户、财务、公告、研报或价格数据。
2. 当前在线数据来自 a-stock-data 适配层，可能包含腾讯行情、东财基本面/板块/新闻/资金流、巨潮公告；接口失败时要说明失败源。
3. 研究结论只能表达“优先研究”“证据强弱”“下一步验证”，不能给买入、卖出、持有、目标价或收益承诺。
4. 如果用户要求 Serenity 或 Bottleneck，要先排产业链层级，再排公司；区分“真实卡点”和“主题受益”。
5. 必须严格区分“运行日期 / 抓取时间 / 数据自身日期”。不要把公告日期、新闻日期、资金流 latest_date 或交易数据日期写成“今日”。如果要写“今日”，只能指运行日期。
6. 对“公司运营情况”的描述必须引用在线数据包里的实际字段，例如 price、change_pct、pe_ttm、pb、mcap、industry、concept_tags、fund_flow、announcements、news；没有字段就写“现有公开抓取数据不足”。
7. 对“关系图谱”的解释只能解释在线数据包和结构化公司数据中出现的公司、板块、概念、公告、新闻、资金流和本地公司映射，不要新增未出现在数据包里的节点或关系。
8. 输出中文，直接、清晰，适合作为投研初筛材料。
"""


REPORT_USER_PROMPT_TEMPLATE = """用户问题：
{question}

检索资料：
{retrieved_context}

结构化公司数据：
{company_context}

请按以下格式输出中文研究结果：

## 1. 核心结论

## 2. 产业链拆解
### 上游
### 中游
### 下游

## 3. A股公司映射表
用 Markdown 表格输出，列为：公司、代码、细分环节、相关逻辑、龙头评分、风险提示。

## 4. 最可能卡住产业发展的环节

## 5. 参考来源
列出检索资料中的 source 和 chunk_index。
"""


def build_report_prompt(question: str, retrieved_context: str, company_context: str) -> str:
    return REPORT_USER_PROMPT_TEMPLATE.format(
        question=question,
        retrieved_context=retrieved_context,
        company_context=company_context,
    )


ONLINE_RESEARCH_PROMPT_TEMPLATE = """用户问题：
{question}

运行日期：
{run_date}

研究模式：
{research_mode}

在线数据包：
{online_context}

结构化公司数据：
{company_context}

请按对应模式输出中文研究结果。

日期要求：
- 第一段必须写明：本次运行日期是 {run_date}。
- 如果引用公告、新闻或资金流日期，必须写成“公告日期/新闻日期/资金流最新日期”，不能写成“今日”。
- 如果在线行情没有明确交易日期，只能写“本次抓取时的行情字段”，不要写“今日交易时段内”。
- 例如 2026-06-05 出现在巨潮公告中时，只能表述为“最近公告日期为 2026-06-05”，不能表述为“今日（2026-06-05）”。

如果 research_mode = bottleneck_hunter：
## 1. 快速结论：最可能卡住的层级
先排层级，不先排股票。列出 3-5 个产业链层级，并说明为什么这些地方更接近真实扩产约束。

## 2. 瓶颈层级排序
用表格：层级 / 为什么可能卡住 / 相关公司或线索 / 证据强度 / 下一步验证。

## 3. 公司初筛
用表格：公司 / 代码 / 它卡住什么或靠近什么 / 在线证据 / 主要风险。

## 4. 证据缺口
列出还需要查的公告、财报、客户认证、订单、产能或资金流证据。

如果 research_mode = serenity：
## 1. 核心判断
用 Serenity 方式说明：市场故事 -> 系统变化 -> 必要零部件 -> 供应链卡点 -> 上市公司 -> 证据 -> 什么情况说明判断错了。

## 2. 产业链层级
按上游/中游/下游或更细层级拆解，并指出稀缺层。

## 3. 优先研究名单
用表格：公司 / 代码 / 产业链位置 / 排序原因 / 在线证据 / 证据强度 / 主要风险。

## 4. 反方理由
说明哪些情况会削弱这个判断。

## 5. 下一步验证
列出具体要查的公开来源。

如果 research_mode = a_stock_online：
## 1. 在线数据摘要
## 2. 公司/标的对比
## 3. 资金面、公告、新闻和概念线索
## 4. 结论和下一步验证

无论哪种模式，最后都要列出“使用的数据源”，包括 source、code、status。
"""


def build_online_research_prompt(
    question: str,
    research_mode: str,
    run_date: str,
    online_context: str,
    company_context: str,
) -> str:
    return ONLINE_RESEARCH_PROMPT_TEMPLATE.format(
        question=question,
        research_mode=research_mode,
        run_date=run_date,
        online_context=online_context,
        company_context=company_context,
    )
