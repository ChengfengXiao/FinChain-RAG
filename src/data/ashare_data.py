from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests


UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
EM_MIN_INTERVAL = 0.8
EASTMONEY_SEARCH_URL = "https://searchapi.eastmoney.com/api/suggest/get"
QUESTION_STOPWORDS = [
    "帮我",
    "请",
    "分析",
    "研究",
    "搜索",
    "查询",
    "一下",
    "怎么样",
    "如何",
    "公司",
    "产业链",
    "产业",
    "行业",
    "相关",
    "有哪些",
    "运营情况",
    "经营情况",
    "图谱",
    "的",
]


def normalize_code(value: str) -> str | None:
    match = re.search(r"(?<!\d)(?:SH|SZ|BJ)?(\d{6})(?:\.(?:SH|SZ|BJ))?(?!\d)", value, re.IGNORECASE)
    if not match:
        return None
    return match.group(1)


def market_prefix(code: str) -> str:
    if code.startswith(("6", "9")):
        return "sh"
    if code.startswith("8"):
        return "bj"
    return "sz"


def eastmoney_market_code(code: str) -> int:
    return 1 if code.startswith("6") else 0


@dataclass
class AShareSource:
    source: str
    code: str
    title: str
    data: Any
    status: str = "ok"
    fetched_at: str = ""


class AShareDataClient:
    """Small runtime adapter for the project-level a-stock-data skill.

    The skill itself is a large Markdown knowledge bundle. This adapter implements
    the most useful zero-key endpoints directly so the app can collect live
    context before calling the LLM.
    """

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({"User-Agent": UA})
        self._last_em_request = 0.0
        self._cninfo_orgid_map: dict[str, str] = {}

    def _em_get(self, url: str, **kwargs: Any) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(1):
            elapsed = time.time() - self._last_em_request
            if elapsed < EM_MIN_INTERVAL:
                time.sleep(EM_MIN_INTERVAL - elapsed)
            try:
                response = self.session.get(url, **kwargs)
                self._last_em_request = time.time()
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(0.5 + attempt)
        if last_error:
            raise last_error
        raise RuntimeError("eastmoney request failed")

    def collect(self, codes: list[str], news_limit: int = 3, announcement_limit: int = 3) -> list[AShareSource]:
        normalized = []
        for code in codes:
            clean = normalize_code(code) or code
            if re.fullmatch(r"\d{6}", clean) and clean not in normalized:
                normalized.append(clean)

        sources: list[AShareSource] = []
        if not normalized:
            return sources

        sources.extend(self._safe_many("tencent_quote", normalized, lambda: self.tencent_quote(normalized)))
        for code in normalized:
            sources.extend(
                [
                    self._safe_one(code, "eastmoney_stock_info", "东财个股基本面", lambda c=code: self.eastmoney_stock_info(c)),
                    self._safe_one(code, "eastmoney_business_analysis", "东财F10经营分析/主营构成", lambda c=code: self.eastmoney_business_analysis(c)),
                    self._safe_one(code, "eastmoney_concept_blocks", "东财概念/行业板块", lambda c=code: self.eastmoney_concept_blocks(c)),
                    self._safe_one(code, "stock_fund_flow_120d", "东财120日资金流摘要", lambda c=code: self.fund_flow_summary(c)),
                    self._safe_one(
                        code,
                        "eastmoney_stock_news",
                        "东财个股新闻",
                        lambda c=code: self.eastmoney_stock_news(c, page_size=news_limit),
                    ),
                    self._safe_one(
                        code,
                        "cninfo_announcements",
                        "巨潮公告",
                        lambda c=code: self.cninfo_announcements(c, page_size=announcement_limit),
                    ),
                ]
            )
        return sources

    def collect_company_ops(self, query: str, limit: int = 1) -> tuple[list[str], list[AShareSource]]:
        codes = []
        explicit = normalize_code(query)
        if explicit:
            codes = [explicit]
            search_sources: list[AShareSource] = []
        else:
            codes, search_sources, _ = self.resolve_targets(query, limit=limit)
        return codes[:limit], [*search_sources, *self.collect(codes[:limit])]

    def resolve_targets(self, query: str, limit: int = 4) -> tuple[list[str], list[AShareSource], dict[str, Any]]:
        """Resolve a user query into A-share stock codes using public Eastmoney search data."""
        candidates = self._search_candidates(query)
        search_records: list[dict[str, Any]] = []
        selected_codes: list[str] = []
        selected_blocks: list[dict[str, Any]] = []

        for candidate in candidates:
            try:
                rows = self.eastmoney_search(candidate, count=max(limit, 8))
            except Exception as exc:
                search_records.append({"candidate": candidate, "status": "error", "error": str(exc)})
                continue
            search_records.append({"candidate": candidate, "results": rows[:8]})
            if not rows:
                continue

            stock_rows = [row for row in rows if row.get("classify") == "AStock" and re.fullmatch(r"\d{6}", row.get("code", ""))]
            block_rows = [row for row in rows if row.get("classify") == "BK" and str(row.get("code", "")).startswith("BK")]

            for row in stock_rows:
                code = row["code"]
                if code not in selected_codes:
                    selected_codes.append(code)
                if len(selected_codes) >= limit:
                    break

            if not selected_codes and block_rows:
                block = block_rows[0]
                selected_blocks.append(block)
                constituents = self.eastmoney_block_constituents(block["code"], limit=limit)
                for item in constituents:
                    code = item.get("code", "")
                    if re.fullmatch(r"\d{6}", code) and code not in selected_codes:
                        selected_codes.append(code)
                if selected_codes:
                    break

            if selected_codes:
                break

        fetched_at = datetime.now().isoformat(timespec="seconds")
        source = AShareSource(
            source="eastmoney_search",
            code=",".join(selected_codes),
            title="东财关键词搜索/板块成分映射",
            data={"query": query, "candidates": candidates, "records": search_records, "selected_blocks": selected_blocks},
            status="ok" if selected_codes else "error",
            fetched_at=fetched_at,
        )
        meta = {"query": query, "candidates": candidates, "selected_blocks": selected_blocks}
        return selected_codes[:limit], [source], meta

    def eastmoney_search(self, keyword: str, count: int = 8) -> list[dict[str, Any]]:
        if not keyword.strip():
            return []
        response = self.session.get(
            EASTMONEY_SEARCH_URL,
            params={"input": keyword.strip(), "type": "14", "token": "123", "count": count},
            timeout=8,
        )
        response.raise_for_status()
        data = ((response.json().get("QuotationCodeTable") or {}).get("Data")) or []
        rows = []
        for item in data:
            rows.append(
                {
                    "code": item.get("Code") or item.get("UnifiedCode") or "",
                    "name": item.get("Name", ""),
                    "classify": item.get("Classify", ""),
                    "security_type_name": item.get("SecurityTypeName", ""),
                    "quote_id": item.get("QuoteID", ""),
                }
            )
        return rows

    def eastmoney_block_constituents(self, block_code: str, limit: int = 10) -> list[dict[str, Any]]:
        response = self._em_get(
            "https://push2.eastmoney.com/api/qt/clist/get",
            params={
                "pn": "1",
                "pz": str(limit),
                "po": "1",
                "np": "1",
                "fltt": "2",
                "invt": "2",
                "fid": "f3",
                "fs": f"b:{block_code}",
                "fields": "f12,f14,f2,f3,f20,f21,f9,f23",
            },
            timeout=8,
        )
        rows = ((response.json().get("data") or {}).get("diff")) or []
        return [
            {
                "code": item.get("f12", ""),
                "name": item.get("f14", ""),
                "price": item.get("f2", ""),
                "change_pct": item.get("f3", ""),
                "mcap_yuan": item.get("f20", ""),
                "float_mcap_yuan": item.get("f21", ""),
                "pe_ttm": item.get("f9", ""),
                "pb": item.get("f23", ""),
            }
            for item in rows
        ]

    def _search_candidates(self, query: str) -> list[str]:
        codes = re.findall(r"(?<!\d)(?:SH|SZ|BJ)?(\d{6})(?:\.(?:SH|SZ|BJ))?(?!\d)", query, re.IGNORECASE)
        if codes:
            return codes

        cleaned = re.sub(r"[，。！？、；：,.!?;:\s]+", "", query)
        for word in QUESTION_STOPWORDS:
            cleaned = cleaned.replace(word, "")
        candidates: list[str] = []
        if cleaned:
            candidates.append(cleaned)

        chinese = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,12}", query)
        for token in chinese:
            token_clean = token
            for word in QUESTION_STOPWORDS:
                token_clean = token_clean.replace(word, "")
            if len(token_clean) >= 2 and token_clean not in candidates:
                candidates.append(token_clean)

        for keyword in ["液冷", "算力", "数据中心", "光模块", "机器人", "半导体", "锂电", "储能", "AI"]:
            if keyword in query and keyword not in candidates:
                candidates.append(keyword)

        # Keep a small deterministic set to avoid hammering public endpoints.
        return candidates[:8]

    def _safe_many(self, source: str, codes: list[str], fn: Any) -> list[AShareSource]:
        fetched_at = datetime.now().isoformat(timespec="seconds")
        try:
            data = fn()
            return [
                AShareSource(
                    source=source,
                    code=code,
                    title="腾讯实时行情/估值",
                    data=data.get(code, {}),
                    fetched_at=fetched_at,
                )
                for code in codes
            ]
        except Exception as exc:
            return [
                AShareSource(
                    source=source,
                    code=",".join(codes),
                    title="腾讯实时行情/估值",
                    data=str(exc),
                    status="error",
                    fetched_at=fetched_at,
                )
            ]

    def _safe_one(self, code: str, source: str, title: str, fn: Any) -> AShareSource:
        fetched_at = datetime.now().isoformat(timespec="seconds")
        try:
            return AShareSource(source=source, code=code, title=title, data=fn(), fetched_at=fetched_at)
        except Exception as exc:
            return AShareSource(source=source, code=code, title=title, data=str(exc), status="error", fetched_at=fetched_at)

    def tencent_quote(self, codes: list[str]) -> dict[str, dict[str, Any]]:
        prefixed = [f"{market_prefix(code)}{code}" for code in codes]
        response = self.session.get("https://qt.gtimg.cn/q=" + ",".join(prefixed), timeout=10)
        response.raise_for_status()
        data = response.content.decode("gbk", errors="ignore")

        result: dict[str, dict[str, Any]] = {}
        for line in data.strip().split(";"):
            if not line.strip() or "=" not in line or '"' not in line:
                continue
            key = line.split("=")[0].split("_")[-1]
            vals = line.split('"')[1].split("~")
            if len(vals) < 53:
                continue
            code = key[2:]
            result[code] = {
                "name": vals[1],
                "price": _to_float(vals[3]),
                "change_pct": _to_float(vals[32]),
                "amount_wan": _to_float(vals[37]),
                "turnover_pct": _to_float(vals[38]),
                "pe_ttm": _to_float(vals[39]),
                "mcap_yi": _to_float(vals[44]),
                "float_mcap_yi": _to_float(vals[45]),
                "pb": _to_float(vals[46]),
                "vol_ratio": _to_float(vals[49]),
                "pe_static": _to_float(vals[52]),
            }
        return result

    def eastmoney_stock_info(self, code: str) -> dict[str, Any]:
        params = {
            "fltt": "2",
            "invt": "2",
            "fields": "f57,f58,f84,f85,f127,f116,f117,f189,f43",
            "secid": f"{eastmoney_market_code(code)}.{code}",
        }
        response = self._em_get("https://push2.eastmoney.com/api/qt/stock/get", params=params, timeout=6)
        data = response.json().get("data") or {}
        return {
            "code": data.get("f57", ""),
            "name": data.get("f58", ""),
            "industry": data.get("f127", ""),
            "total_shares": data.get("f84", 0),
            "float_shares": data.get("f85", 0),
            "mcap_yuan": data.get("f116", 0),
            "float_mcap_yuan": data.get("f117", 0),
            "list_date": str(data.get("f189", "")),
            "price_raw": data.get("f43", 0),
        }

    def eastmoney_business_analysis(self, code: str) -> dict[str, Any]:
        market = market_prefix(code).upper()
        response = self.session.get(
            "https://emweb.securities.eastmoney.com/PC_HSF10/BusinessAnalysis/PageAjax",
            params={"code": f"{market}{code}"},
            headers={"User-Agent": UA, "Referer": "https://emweb.securities.eastmoney.com/"},
            timeout=10,
        )
        response.raise_for_status()
        raw = response.json()
        scope_rows = raw.get("zyfw") or []
        composition_rows = raw.get("zygcfx") or []
        review_rows = raw.get("jyps") or []
        latest_date = ""
        for row in composition_rows:
            date = str(row.get("REPORT_DATE", ""))[:10]
            if date > latest_date:
                latest_date = date

        rows = []
        for row in composition_rows:
            if latest_date and str(row.get("REPORT_DATE", ""))[:10] != latest_date:
                continue
            rows.append(
                {
                    "report_date": str(row.get("REPORT_DATE", ""))[:10],
                    "type": _mainop_type_name(str(row.get("MAINOP_TYPE", ""))),
                    "item_name": row.get("ITEM_NAME", ""),
                    "revenue_yuan": _to_float(row.get("MAIN_BUSINESS_INCOME")),
                    "revenue_ratio": _to_float(row.get("MBI_RATIO")),
                    "cost_yuan": _to_float(row.get("MAIN_BUSINESS_COST")),
                    "cost_ratio": _to_float(row.get("MBC_RATIO")),
                    "profit_yuan": _to_float(row.get("MAIN_BUSINESS_RPOFIT")),
                    "profit_ratio": _to_float(row.get("MBR_RATIO")),
                    "gross_margin": _to_float(row.get("GROSS_RPOFIT_RATIO")),
                }
            )

        return {
            "business_scope": (scope_rows[0].get("BUSINESS_SCOPE", "") if scope_rows else ""),
            "business_review": _strip_html(review_rows[0].get("BUSINESS_REVIEW", ""))[:1200] if review_rows else "",
            "latest_report_date": latest_date,
            "composition": rows,
        }

    def eastmoney_concept_blocks(self, code: str) -> dict[str, Any]:
        params = {
            "fltt": "2",
            "invt": "2",
            "secid": f"{eastmoney_market_code(code)}.{code}",
            "spt": "3",
            "pi": "0",
            "pz": "200",
            "po": "1",
            "fields": "f12,f14,f3,f128",
        }
        headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}
        response = self._em_get("https://push2.eastmoney.com/api/qt/slist/get", params=params, headers=headers, timeout=8)
        diff = ((response.json().get("data") or {}).get("diff")) or {}
        items = diff.values() if isinstance(diff, dict) else diff
        boards = [
            {
                "name": item.get("f14", ""),
                "code": item.get("f12", ""),
                "change_pct": item.get("f3", ""),
                "lead_stock": item.get("f128", ""),
            }
            for item in items
        ]
        return {"total": len(boards), "concept_tags": [item["name"] for item in boards], "boards": boards[:30]}

    def stock_fund_flow_120d(self, code: str) -> list[dict[str, Any]]:
        params = {
            "secid": f"{eastmoney_market_code(code)}.{code}",
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
            "lmt": "120",
        }
        headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/", "Origin": "https://quote.eastmoney.com"}
        response = self._em_get(
            "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get",
            params=params,
            headers=headers,
            timeout=8,
        )
        klines = ((response.json().get("data") or {}).get("klines")) or []
        rows = []
        for line in klines:
            parts = line.split(",")
            if len(parts) >= 6:
                rows.append(
                    {
                        "date": parts[0],
                        "main_net": _to_float(parts[1]),
                        "small_net": _to_float(parts[2]),
                        "mid_net": _to_float(parts[3]),
                        "large_net": _to_float(parts[4]),
                        "super_net": _to_float(parts[5]),
                    }
                )
        return rows

    def fund_flow_summary(self, code: str) -> dict[str, Any]:
        rows = self.stock_fund_flow_120d(code)
        recent = rows[-20:]
        last5 = rows[-5:]
        return {
            "latest_date": rows[-1]["date"] if rows else "",
            "main_net_5d_yuan": sum(row["main_net"] for row in last5),
            "main_net_20d_yuan": sum(row["main_net"] for row in recent),
            "super_net_5d_yuan": sum(row["super_net"] for row in last5),
            "large_net_5d_yuan": sum(row["large_net"] for row in last5),
            "last_5_days": last5,
        }

    def eastmoney_stock_news(self, code: str, page_size: int = 5) -> list[dict[str, Any]]:
        callback = "jQuery_news"
        inner_params = json.dumps(
            {
                "uid": "",
                "keyword": code,
                "type": ["cmsArticleWebOld"],
                "client": "web",
                "clientType": "web",
                "clientVersion": "curr",
                "param": {
                    "cmsArticleWebOld": {
                        "searchScope": "default",
                        "sort": "default",
                        "pageIndex": 1,
                        "pageSize": page_size,
                        "preTag": "",
                        "postTag": "",
                    }
                },
            },
            separators=(",", ":"),
        )
        headers = {"User-Agent": UA, "Referer": "https://so.eastmoney.com/"}
        response = self._em_get(
            "https://search-api-web.eastmoney.com/search/jsonp",
            params={"cb": callback, "param": inner_params},
            headers=headers,
            timeout=8,
        )
        text = response.text
        json_text = text[text.index("(") + 1 : text.rindex(")")]
        data = json.loads(json_text)
        articles = ((data.get("result") or {}).get("cmsArticleWebOld")) or []
        return [
            {
                "title": _strip_html(item.get("title", "")),
                "content": _strip_html(item.get("content", ""))[:160],
                "time": item.get("date", ""),
                "source": item.get("mediaName", ""),
                "url": item.get("url", ""),
            }
            for item in articles[:page_size]
        ]

    def _cninfo_orgid(self, code: str) -> str:
        if not self._cninfo_orgid_map:
            try:
                response = self.session.get("http://www.cninfo.com.cn/new/data/szse_stock.json", timeout=8)
                response.raise_for_status()
                self._cninfo_orgid_map = {
                    item["code"]: item["orgId"] for item in response.json().get("stockList", [])
                }
            except Exception:
                self._cninfo_orgid_map = {}
        org_id = self._cninfo_orgid_map.get(code)
        if org_id:
            return org_id
        if code.startswith("6"):
            return f"gssh0{code}"
        if code.startswith(("8", "4")):
            return f"gsbj0{code}"
        return f"gssz0{code}"

    def cninfo_announcements(self, code: str, page_size: int = 5) -> list[dict[str, Any]]:
        payload = {
            "stock": f"{code},{self._cninfo_orgid(code)}",
            "tabName": "fulltext",
            "pageSize": str(page_size),
            "pageNum": "1",
            "column": "",
            "category": "",
            "plate": "",
            "seDate": "",
            "searchkey": "",
            "secid": "",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        headers = {
            "User-Agent": UA,
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": "https://www.cninfo.com.cn/new/disclosure",
            "Origin": "https://www.cninfo.com.cn",
        }
        response = self.session.post("https://www.cninfo.com.cn/new/hisAnnouncement/query", data=payload, headers=headers, timeout=8)
        response.raise_for_status()
        rows = []
        for item in response.json().get("announcements", []) or []:
            rows.append(
                {
                    "title": _strip_html(item.get("announcementTitle", "")),
                    "type": item.get("announcementTypeName", ""),
                    "date": _cninfo_ts_to_date(item.get("announcementTime")),
                    "url": f"https://www.cninfo.com.cn/new/disclosure/detail?annoId={item.get('announcementId', '')}",
                }
            )
        return rows[:page_size]


def _to_float(value: Any) -> float:
    try:
        if value in {"", "-", None}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value or "")


def _mainop_type_name(value: str) -> str:
    return {"1": "按产品", "2": "按行业", "3": "按地区"}.get(value, value or "未知")


def _cninfo_ts_to_date(value: Any) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000).strftime("%Y-%m-%d")
    return str(value)[:10] if value else ""
