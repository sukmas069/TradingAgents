"""A-share news and exchange announcements, and China's domestic headlines.

Company news comes from East Money's per-stock feed, which serves only its
latest items. Announcements come from CNINFO, the exchanges' designated
disclosure site, which is queried by date range and so serves any past window
in full. Domestic headlines come from the CCTV evening news (by date) and the
Cailian Press wire (latest items only).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

import pandas as pd

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import coverage_gap, in_window
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.vendors.china.common import (
    CNINFO,
    EAST_MONEY,
    ak_call,
    akshare,
    cached_frame,
    require_a_share,
    ymd,
)

logger = logging.getLogger(__name__)

# Days of CCTV evening news read back from the analysis date. Each day is its
# own slow request, and the programme's agenda moves slowly.
_CCTV_DAYS = 3

# China Standard Time: the Chinese feeds stamp items in local time, without an offset.
_CST = "+08:00"

# Announcement titles that bear on risk or ownership, for the fundamentals and
# insider views. Matched as substrings of the Chinese title.
RISK_KEYWORDS = (
    "风险警示", "退市", "撤销", "立案", "处罚", "监管", "问询", "关注函", "警示函",
    "业绩预告", "业绩快报", "预亏", "亏损", "减值", "诉讼", "仲裁", "冻结", "质押",
    "担保", "违规", "审计", "保留意见", "无法表示意见", "非标", "重组", "停牌", "复牌",
)
OWNERSHIP_KEYWORDS = ("增持", "减持", "持股", "股份变动", "权益变动", "回购", "解除限售", "限售股")


def _cst(text) -> datetime | None:
    """A Chinese feed's local timestamp (``2026-10-12 19:04:29``) as an aware datetime."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    raw = str(text).strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("/", "-") + ("" if "+" in raw else _CST))
    except ValueError:
        return None


def fetch_announcements(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """CNINFO announcements for ``ticker`` published in ``[start_date, end_date]``.

    Columns: ``time`` (aware datetime), ``title``, ``link``; newest first.
    """
    code, _ = require_a_share(ticker)
    ak = akshare()
    frame = ak_call(
        ak.stock_zh_a_disclosure_report_cninfo, symbol=code, market="沪深京",
        start_date=ymd(start_date), end_date=ymd(end_date),
        what=f"{CNINFO} announcements for {code}",
    )
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["time", "title", "link"])
    out = pd.DataFrame({
        "time": frame["公告时间"].map(_cst),
        "title": frame["公告标题"].astype(str),
        "link": frame.get("公告链接", pd.Series([""] * len(frame))).astype(str),
    })
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    # CNINFO filters by date already; trimming again keeps a loose upstream
    # filter from ever leaking a later announcement into a backtest.
    out = out[out["time"].map(lambda t: in_window(t, start_dt, end_dt))]
    return out.sort_values("time", ascending=False).reset_index(drop=True)


def format_announcements(frame: pd.DataFrame, limit: int, keywords: tuple[str, ...] | None = None) -> str:
    """Announcement titles as a dated list, optionally only those matching ``keywords``."""
    if keywords:
        pattern = "|".join(map(re.escape, keywords))
        frame = frame[frame["title"].str.contains(pattern, regex=True)]
    lines = [f"- [{t:%Y-%m-%d %H:%M}] {title}" for t, title in zip(frame["time"], frame["title"], strict=True)]
    shown = lines[:limit]
    more = f"\n- … {len(lines) - limit} more not shown" if len(lines) > limit else ""
    return "\n".join(shown) + more if shown else ""


def _company_news(code: str) -> pd.DataFrame:
    ak = akshare()
    frame = ak_call(ak.stock_news_em, symbol=code, what=f"{EAST_MONEY} news for {code}")
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["time", "title", "summary", "source", "link"])
    return pd.DataFrame({
        "time": frame["发布时间"].map(_cst),
        "title": frame["新闻标题"].astype(str),
        "summary": frame["新闻内容"].astype(str),
        "source": frame["文章来源"].astype(str),
        "link": frame["新闻链接"].astype(str),
    })


def get_news_cn(ticker: str, start_date: str, end_date: str) -> str:
    """Company news and exchange announcements published in the window.

    Announcements are the primary disclosure channel for A-shares (results,
    risk warnings, regulatory letters, major contracts), so both are served.
    East Money's news feed holds only its latest items; a window it cannot
    reach is reported as unavailable, while CNINFO answers for any window.
    """
    code, _ = require_a_share(ticker)
    limit = get_config()["news_article_limit"]
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    sections = [f"## {ticker.upper()} news and announcements, from {start_date} to {end_date}"]

    try:
        news = _company_news(code)
        in_range = news[news["time"].map(lambda t: in_window(t, start_dt, end_dt))]
        if not in_range.empty:
            items = []
            for row in in_range.head(limit).itertuples():
                summary = row.summary.strip().replace("\n", " ")
                if len(summary) > 300:
                    summary = summary[:300] + "…"
                items.append(f"### {row.title} (source: {row.source}, {row.time:%Y-%m-%d %H:%M})\n{summary}\n")
            sections.append(f"### Company news — {EAST_MONEY}\n\n" + "\n".join(items))
        else:
            gap = coverage_gap(news["time"].tolist(), start_date, end_date,
                               f"{EAST_MONEY} news", f"news about {ticker}")
            sections.append(f"### Company news — {EAST_MONEY}\n\n"
                            + (gap or f"<no {EAST_MONEY} news about {ticker} in {start_date}..{end_date}>"))
    except VendorUnavailableError as exc:
        logger.warning("East Money news unavailable for %s: %s", ticker, exc)
        sections.append(f"### Company news — {EAST_MONEY}\n\n<unavailable: {exc}>")

    # Announcements over a wider window than the news: a risk warning or a
    # regulatory letter from last month still frames this week.
    ann_start = min(start_dt, end_dt - timedelta(days=30)).strftime("%Y-%m-%d")
    announcements = fetch_announcements(ticker, ann_start, end_date)
    listed = format_announcements(announcements, limit=40)
    sections.append(
        f"### Exchange announcements — {CNINFO}, {ann_start} to {end_date}\n\n"
        + (listed or f"<no announcements filed in {ann_start}..{end_date}>")
    )
    return "\n\n".join(sections)


def get_global_news_cn(as_of_date: str, look_back_days: int | None = None, limit: int | None = None) -> str:
    """China's domestic headlines: CCTV evening news by day, and the Cailian Press wire.

    The CCTV programme is fetched per date, so any past window is served as it
    aired. The Cailian wire serves its latest items, so a past window it cannot
    reach is reported as unavailable.
    """
    config = get_config()
    look_back_days = look_back_days or config["global_news_lookback_days"]
    limit = limit or config["global_news_article_limit"]
    ak = akshare()
    end_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    sections = [f"## China domestic news, from {start_date} to {as_of_date}"]

    cctv_lines = []
    for offset in range(min(look_back_days, _CCTV_DAYS)):
        day = end_dt - timedelta(days=offset)
        stamp = day.strftime("%Y%m%d")
        try:
            frame = cached_frame("cctv", stamp, lambda stamp=stamp: ak_call(
                ak.news_cctv, date=stamp, what=f"CCTV news for {stamp}"))
        except VendorUnavailableError as exc:
            logger.warning("%s", exc)
            continue
        if frame is None or frame.empty:
            continue
        titles = [str(t) for t in frame["title"].tolist()][:6]
        cctv_lines.append(f"- {day:%Y-%m-%d}: " + "；".join(titles))
    sections.append("### CCTV evening news (新闻联播) headlines\n\n"
                    + ("\n".join(cctv_lines) or "<CCTV evening news unavailable for this window>"))

    try:
        wire = ak_call(ak.stock_info_global_cls, symbol="全部", what="Cailian Press wire")
        times = [_cst(f"{d} {t}") for d, t in zip(wire["发布日期"], wire["发布时间"], strict=True)]
        rows = [(t, title, body) for t, title, body in zip(times, wire["标题"], wire["内容"], strict=True)
                if in_window(t, start_dt, end_dt)]
        rows.sort(key=lambda row: row[0], reverse=True)
        if rows:
            items = []
            for t, title, body in rows[:limit]:
                text = str(body).strip().replace("\n", " ")
                headline = str(title).strip() if pd.notna(title) and str(title).strip() else text[:80]
                items.append(f"- [{t:%Y-%m-%d %H:%M}] {headline}: {text[:240]}")
            sections.append("### Cailian Press wire (财联社电报)\n\n" + "\n".join(items))
        else:
            gap = coverage_gap(times, start_date, as_of_date, "Cailian Press wire", "China market news")
            sections.append("### Cailian Press wire (财联社电报)\n\n"
                            + (gap or f"<no Cailian Press items in {start_date}..{as_of_date}>"))
    except VendorUnavailableError as exc:
        sections.append(f"### Cailian Press wire (财联社电报)\n\n<unavailable: {exc}>")

    if not cctv_lines and "unavailable" in sections[-1]:
        raise VendorUnavailableError("no China domestic news source answered")
    return "\n\n".join(sections)


def get_insider_transactions_cn(ticker: str, as_of_date: str | None = None) -> str:
    """Shareholding-change announcements (增减持、回购、解禁) from the past 180 days.

    A-share insiders and major holders disclose trades and planned trades
    through exchange announcements, which CNINFO dates by publication, so a
    past run sees exactly what had been disclosed by then.
    """
    require_a_share(ticker)
    end = as_of_date or datetime.now().strftime("%Y-%m-%d")
    start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=180)).strftime("%Y-%m-%d")
    frame = fetch_announcements(ticker, start, end)
    listed = format_announcements(frame, limit=30, keywords=OWNERSHIP_KEYWORDS)
    header = (f"# Shareholding-change announcements for {ticker.upper()}, {start} to {end}\n"
              f"# Source: {CNINFO}. Titles only; trade sizes and prices are in the announcement text.\n\n")
    return header + (listed or "No shareholding-change announcements (增持/减持/回购/解禁) in this window.")
