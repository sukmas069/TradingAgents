"""A-share retail chatter: the East Money stock forum (股吧) and its popularity ranking.

The forum is the largest Chinese retail board for individual stocks. Its list
page embeds the latest posts as JSON, with publish time, clicks and replies, so
posts are trimmed to the analysis window like StockTwits messages; the page
serves only recent posts, so a past window it cannot reach is reported as
unavailable. East Money's popularity rank (人气榜) is a daily history, served
up to the analysis date.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import pandas as pd
import requests

from tradingagents.dataflows.date_window import coverage_gap, in_window
from tradingagents.dataflows.vendors.china.common import (
    EAST_MONEY,
    ak_call,
    akshare,
    require_a_share,
)
from tradingagents.dataflows.vendors.china.news import _cst

logger = logging.getLogger(__name__)

_GUBA_LIST = "https://guba.eastmoney.com/list,{code},f_{page}.html"
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
_MARKER = "var article_list="


def _guba_page(code: str, page: int, timeout: float) -> list[dict]:
    """The posts embedded in one forum list page, newest first."""
    response = requests.get(_GUBA_LIST.format(code=code, page=page), headers={"User-Agent": _UA}, timeout=timeout)
    response.raise_for_status()
    text = response.text
    start = text.find(_MARKER)
    if start < 0:
        raise ValueError("forum page carries no post list")
    data, _ = json.JSONDecoder().raw_decode(text[start + len(_MARKER):])
    return data.get("re") or []


def fetch_guba_posts(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 40,
    pages: int = 2,
    timeout: float = 10.0,
) -> str:
    """Recent forum posts about ``ticker`` as a prompt-ready block.

    Pinned and official posts (announcement mirrors, ads) are skipped, since
    they are not retail opinion. Returns a placeholder instead of raising, like
    the other social fetchers, so the sentiment analyst always gets a block.
    """
    try:
        code, _ = require_a_share(ticker)
    except Exception as exc:  # noqa: BLE001
        return f"<East Money forum unavailable: {exc}>"

    posts: list[dict] = []
    try:
        for page in range(1, pages + 1):
            posts += _guba_page(code, page, timeout)
    except (requests.RequestException, ValueError) as exc:
        logger.warning("East Money forum fetch failed for %s: %s", ticker, exc)
        if not posts:
            return f"<East Money forum unavailable: {type(exc).__name__}>"

    retail = [p for p in posts if not p.get("post_top_status") and p.get("post_type", 0) == 0]
    stamped = [(_cst(p.get("post_publish_time")), p) for p in retail]
    if start_date and end_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        kept = [(t, p) for t, p in stamped if in_window(t, start_dt, end_dt)]
        if not kept:
            gap = coverage_gap((t for t, _ in stamped), start_date, end_date,
                               "East Money forum", f"posts about {ticker}")
            return gap or f"<no East Money forum posts about {ticker} within {start_date}..{end_date}>"
    else:
        kept = stamped

    lines = []
    clicks = replies = 0
    for t, post in kept[:limit]:
        title = str(post.get("post_title") or "").replace("\n", " ").strip()
        click, reply = int(post.get("post_click_count") or 0), int(post.get("post_comment_count") or 0)
        clicks += click
        replies += reply
        when = f"{t:%Y-%m-%d %H:%M}" if t else "undated"
        lines.append(f"[{when} · {click} views · {reply} replies] {title}")
    summary = (f"{len(lines)} most-recent retail posts shown (of {len(kept)} in the window); "
               f"{clicks} views and {replies} replies in total. Titles only; no user sentiment labels.")
    return summary + "\n\n" + "\n".join(lines)


def fetch_popularity_rank(ticker: str, as_of_date: str, days: int = 20) -> str:
    """East Money's daily popularity rank (人气榜) for the days up to ``as_of_date``.

    Rank 1 is the most-watched stock on East Money that day; a sharp climb is
    retail attention arriving. Placeholder instead of raising.
    """
    try:
        code, exchange = require_a_share(ticker)
        ak = akshare()
        frame = ak_call(ak.stock_hot_rank_detail_em, symbol=f"{exchange}{code}",
                        what=f"{EAST_MONEY} popularity rank for {code}")
    except Exception as exc:  # noqa: BLE001
        return f"<East Money popularity rank unavailable: {exc}>"
    if frame is None or frame.empty:
        return "<East Money popularity rank unavailable: empty response>"
    frame = frame.copy()
    frame["时间"] = pd.to_datetime(frame["时间"], errors="coerce")
    frame = frame[frame["时间"] <= pd.to_datetime(as_of_date)].sort_values("时间").tail(days)
    if frame.empty:
        return f"<East Money popularity rank has no history on or before {as_of_date}>"
    lines = [f"{d:%Y-%m-%d}: rank {int(r)}" for d, r in zip(frame["时间"], frame["排名"], strict=True)]
    first, last = int(frame["排名"].iloc[0]), int(frame["排名"].iloc[-1])
    trend = "rising attention" if last < first else "falling attention" if last > first else "flat"
    return (f"Daily rank among all A-shares (1 = most watched), last {len(lines)} days: "
            f"{first} → {last} ({trend}).\n" + "\n".join(lines))

