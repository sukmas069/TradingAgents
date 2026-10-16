"""A-share daily prices from East Money, with the indicators the market analyst reads."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors.china.common import (
    EAST_MONEY,
    ak_call,
    akshare,
    board_of,
    cached_frame,
    require_a_share,
)
from tradingagents.dataflows.vendors.yahoo.market import get_stock_stats_indicators_window
from tradingagents.dataflows.vendors.yahoo.ohlcv import _assert_ohlcv_not_stale, _fill_price_gaps

# East Money's column names, and the names the indicator code expects.
_COLUMNS = {
    "日期": "Date", "开盘": "Open", "最高": "High", "最低": "Low", "收盘": "Close",
    "成交量": "Volume", "成交额": "Amount", "换手率": "Turnover %", "涨跌幅": "Change %",
}

# Years of history fetched once per symbol per day: enough for a 200-day average
# at any date a backtest is likely to ask about.
_HISTORY_YEARS = 5


def _download(code: str) -> pd.DataFrame:
    ak = akshare()
    end = datetime.now()
    start = end.replace(year=end.year - _HISTORY_YEARS)
    return ak_call(
        ak.stock_zh_a_hist, symbol=code, period="daily",
        start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"),
        adjust="qfq", what=f"{EAST_MONEY} daily prices for {code}",
    )


def load_ohlcv_cn(symbol: str, as_of_date: str, fill_gaps: bool = True) -> pd.DataFrame:
    """Forward-adjusted daily OHLCV for an A-share, with nothing after ``as_of_date``.

    Same frame shape as Yahoo's ``load_ohlcv`` (``Date`` plus capitalised price
    columns), so the indicator and snapshot code reads either. Volume is in
    shares: East Money reports lots of 100. A run dated today refetches, since
    today's bar is partial until the close; an earlier date reuses the day's
    download.
    """
    code, _ = require_a_share(symbol)
    as_of = pd.to_datetime(as_of_date).normalize()
    if as_of.date() >= datetime.now().date():
        raw = _download(code)
    else:
        raw = cached_frame("ohlcv", code, lambda: _download(code))
    if raw is None or raw.empty:
        raise NoMarketDataError(symbol, code, f"{EAST_MONEY} returned no daily prices")

    data = raw.rename(columns=_COLUMNS)
    data = data[[c for c in _COLUMNS.values() if c in data.columns]].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["Date"])
    numeric = [c for c in data.columns if c != "Date"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data["Volume"] = data["Volume"] * 100

    # Point in time: nothing after the analysis date reaches the agents.
    data = data[data["Date"] <= as_of].sort_values("Date").reset_index(drop=True)
    if data.empty:
        raise NoMarketDataError(symbol, code, f"no daily prices on or before {as_of_date}")
    data = _fill_price_gaps(data) if fill_gaps else data.dropna(subset=["Close"]).copy()
    _assert_ohlcv_not_stale(data, as_of_date, symbol, code)
    return data


def get_stock_data_cn(symbol: str, start_date: str, end_date: str) -> str:
    """Daily bars for ``[start_date, end_date]``, with the board's price-limit rule."""
    code, exchange = require_a_share(symbol)
    data = load_ohlcv_cn(symbol, end_date, fill_gaps=False)
    data = data[data["Date"] >= pd.to_datetime(start_date)]
    if data.empty:
        raise NoMarketDataError(symbol, code, f"no daily prices between {start_date} and {end_date}")
    out = data.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    for col in ("Open", "High", "Low", "Close"):
        out[col] = out[col].round(2)
    board, limit = board_of(code, exchange)
    header = (
        f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
        f"# Source: {EAST_MONEY}, forward-adjusted (前复权) daily bars, prices in CNY\n"
        f"# Board: {board}; daily price limit {limit}\n"
        f"# Volume in shares; Amount in CNY; Turnover % is volume over free-float shares\n"
        f"# Total records: {len(out)}\n\n"
    )
    return header + out.to_csv(index=False)


def get_indicators_cn(symbol: str, indicator: str, as_of_date: str, look_back_days: int) -> str:
    """One technical indicator over the window, computed from East Money's bars."""
    require_a_share(symbol)
    return get_stock_stats_indicators_window(
        symbol, indicator, as_of_date, look_back_days, loader=load_ohlcv_cn
    )
