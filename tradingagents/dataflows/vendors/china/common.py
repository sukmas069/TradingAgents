"""What every China-vendor request shares: the AKShare import, the call wrapper, the cache."""

from __future__ import annotations

import logging
import os
from datetime import datetime

import pandas as pd

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorUnavailableError,
)
from tradingagents.dataflows.files import replace_file
from tradingagents.dataflows.symbols import a_share_parts, safe_ticker_component

logger = logging.getLogger(__name__)

# Shown in report headers so a reader can tell where a figure came from.
EAST_MONEY = "East Money (东方财富)"
CNINFO = "CNINFO (巨潮资讯)"


def akshare():
    """The AKShare module, or VendorNotConfiguredError when it is not installed.

    Imported lazily so a missing install only disables this vendor: the router
    then moves on to the next one in the chain.
    """
    try:
        import akshare as ak
    except ImportError as exc:
        raise VendorNotConfiguredError(
            "akshare is not installed; install it with `pip install akshare` "
            "to read China A-share data from Chinese sources"
        ) from exc
    return ak


def ak_call(func, *args, what: str, **kwargs):
    """Call an AKShare function, reporting any failure as an outage.

    AKShare raises whatever its HTTP or parsing layer raised (KeyError on a
    changed page, JSONDecodeError on a blocked request, ...), none of which says
    anything about the instrument, so all of them become VendorUnavailableError
    and the router tries the next vendor.
    """
    try:
        return func(*args, **kwargs)
    except VendorNotConfiguredError:
        raise
    except Exception as exc:  # noqa: BLE001 — a scraper's failure modes are open-ended
        raise VendorUnavailableError(f"{what} could not be retrieved: {type(exc).__name__}: {exc}") from exc


def require_a_share(ticker: str) -> tuple[str, str]:
    """``(code, exchange)`` for an A-share ticker, else NoMarketDataError.

    Raised as "no data" so a non-A-share ticker routed here explicitly falls
    through to the next vendor instead of ending the run.
    """
    parts = a_share_parts(ticker)
    if parts is None:
        raise NoMarketDataError(ticker, ticker, "not a China A-share ticker (expected e.g. 600519.SS or 300010.SZ)")
    return parts


def board_of(code: str, exchange: str) -> tuple[str, str]:
    """The listing board and its daily price-limit rule, from the code prefix."""
    if exchange == "BJ":
        return "Beijing Stock Exchange (北交所)", "±30%"
    if code.startswith("688"):
        return "STAR Market (科创板)", "±20%"
    if code.startswith(("300", "301")):
        return "ChiNext (创业板)", "±20%"
    if exchange == "SH":
        return "Shanghai Main Board (沪市主板)", "±10% (risk-warning ST stocks may differ)"
    return "Shenzhen Main Board (深市主板)", "±10% (risk-warning ST stocks may differ)"


def _cache_path(kind: str, key: str) -> str:
    directory = os.path.join(get_config()["data_cache_dir"], "china")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{safe_ticker_component(key, max_len=64)}-{kind}.pkl")


def cached_frame(kind: str, key: str, fetch) -> pd.DataFrame:
    """``fetch()``'s frame, reused for the rest of the day it was fetched.

    The statement and valuation endpoints take tens of seconds, and every tool
    call in a run asks for the same frame, so one fetch per key per day serves
    them all. Only a non-empty frame is cached.
    """
    path = _cache_path(kind, key)
    if os.path.exists(path):
        written = datetime.fromtimestamp(os.path.getmtime(path)).date()
        if written == datetime.now().date():
            try:
                return pd.read_pickle(path)
            except Exception as exc:  # noqa: BLE001 — a torn cache file is a miss
                logger.warning("Ignoring unreadable cache %s: %s", path, exc)
    frame = fetch()
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        replace_file(path, lambda temp: frame.to_pickle(temp))
    return frame


def ymd(date: str) -> str:
    """``YYYY-MM-DD`` as AKShare's ``YYYYMMDD``."""
    return datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")


def table(frame: pd.DataFrame) -> str:
    """A frame as a Markdown table, or CSV where ``tabulate`` is not installed."""
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_csv(index=False)
