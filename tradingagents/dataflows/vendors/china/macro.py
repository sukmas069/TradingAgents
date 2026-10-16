"""China macro series: CPI, PPI, PMI, money supply, LPR, GDP and SHIBOR.

The tables carry each observation's reference period, not when it was
published and not any later revision. A monthly figure is published some days
after its month ends, so each series holds an observation back until a
conservative publication lag has passed; a backtest then sees roughly what had
been released by its date. Revisions are not tracked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors.china.common import ak_call, akshare, table


@dataclass(frozen=True)
class _Series:
    title: str
    func: str
    period_col: str
    value_cols: tuple[tuple[str, str], ...]   # (source column, label)
    units: str
    # Days after the reference period ends before the figure is treated as public.
    lag_days: int
    period: str = "month"                     # month, quarter-cumulative or day


_SERIES = {
    "cn_cpi": _Series("China CPI (全国居民消费价格指数)", "macro_china_cpi", "月份",
                      (("全国-同比增长", "YoY %"), ("全国-环比增长", "MoM %")), "percent", 15),
    "cn_ppi": _Series("China PPI (工业生产者出厂价格指数)", "macro_china_ppi", "月份",
                      (("当月同比增长", "YoY %"),), "percent", 15),
    "cn_pmi": _Series("China official PMI (官方制造业/非制造业PMI)", "macro_china_pmi", "月份",
                      (("制造业-指数", "Manufacturing"), ("非制造业-指数", "Non-manufacturing")), "index, 50 = no change", 1),
    "cn_m2": _Series("China money supply (货币供应量)", "macro_china_money_supply", "月份",
                     (("货币和准货币(M2)-同比增长", "M2 YoY %"), ("货币(M1)-同比增长", "M1 YoY %")), "percent", 15),
    "cn_lpr": _Series("China Loan Prime Rate (贷款市场报价利率)", "macro_china_lpr", "TRADE_DATE",
                      (("LPR1Y", "1-year LPR %"), ("LPR5Y", "5-year LPR %")), "percent", 0, "day"),
    "cn_gdp": _Series("China GDP growth (国内生产总值, cumulative YoY)", "macro_china_gdp", "季度",
                      (("国内生产总值-同比增长", "GDP YoY %"),), "percent", 20, "quarter"),
    "cn_shibor": _Series("SHIBOR (上海银行间同业拆放利率)", "macro_china_shibor_all", "日期",
                         (("O/N-定价", "Overnight %"), ("1W-定价", "1-week %"), ("3M-定价", "3-month %")),
                         "percent", 0, "day"),
}

ALIASES = tuple(_SERIES)


def _period_end(label, kind: str) -> pd.Timestamp | None:
    """The last day of the period a row describes."""
    text = str(label)
    if kind == "day":
        return pd.to_datetime(text, errors="coerce")
    if kind == "month":
        match = re.match(r"(\d{4})年(\d{1,2})月", text)
        if not match:
            return None
        return pd.Timestamp(int(match[1]), int(match[2]), 1) + pd.offsets.MonthEnd(0)
    match = re.match(r"(\d{4})年第(?:\d-)?(\d)季度", text)
    if not match:
        return None
    return pd.Timestamp(int(match[1]), int(match[2]) * 3, 1) + pd.offsets.MonthEnd(0)


def get_macro_data_cn(indicator: str, as_of_date: str, look_back_days: int | None = None) -> str:
    """A China macro series up to what had been published by ``as_of_date``.

    An alias this vendor does not know raises NoMarketDataError, so the router
    hands US and euro-area series on to FRED.
    """
    key = (indicator or "").strip().lower()
    series = _SERIES.get(key)
    if series is None:
        raise NoMarketDataError(indicator, indicator, f"not a China macro alias (known: {', '.join(ALIASES)})")

    ak = akshare()
    frame = ak_call(getattr(ak, series.func), what=series.title)
    if frame is None or frame.empty:
        raise NoMarketDataError(indicator, key, "empty response")

    frame = frame.copy()
    frame["_end"] = frame[series.period_col].map(lambda v: _period_end(v, series.period))
    frame = frame.dropna(subset=["_end"])
    cutoff = pd.to_datetime(as_of_date) - timedelta(days=series.lag_days)
    frame = frame[frame["_end"] <= cutoff].sort_values("_end")
    window = look_back_days or 365
    start = pd.to_datetime(as_of_date) - timedelta(days=window)
    recent = frame[frame["_end"] >= start]
    if recent.empty:
        recent = frame.tail(1)
    if recent.empty:
        raise NoMarketDataError(indicator, key, f"no observation published by {as_of_date}")

    cols = [(src, label) for src, label in series.value_cols if src in recent.columns]
    shown = recent[[series.period_col] + [src for src, _ in cols]].rename(
        columns={series.period_col: "Period", **dict(cols)})
    if series.period == "day" and len(shown) > 30:
        shown = shown.tail(30)
    shown["Period"] = shown["Period"].astype(str).str.slice(0, 12)

    latest = recent.iloc[-1]
    first = recent.iloc[0]
    summary = []
    for src, label in cols:
        now, then = pd.to_numeric(latest[src], errors="coerce"), pd.to_numeric(first[src], errors="coerce")
        change = "" if pd.isna(now) or pd.isna(then) else f" (change over window: {now - then:+.2f})"
        summary.append(f"- Latest {label}: {now}{change}")

    return (
        f"## {series.title}\n"
        f"- Units: {series.units}\n"
        f"- Point in time: observations published by {as_of_date} "
        f"(each held back {series.lag_days} days after its period ends; later revisions not tracked)\n"
        f"- Latest period: {latest[series.period_col]}\n" + "\n".join(summary)
        + "\n\n" + table(shown.iloc[::-1])
    )


def is_china_macro_alias(indicator) -> bool:
    return isinstance(indicator, str) and indicator.strip().lower() in _SERIES
