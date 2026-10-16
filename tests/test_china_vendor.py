"""The China A-share vendor: routing, and point-in-time integrity on every dated path.

AKShare is replaced by a stub, so these run offline. Each test pins one rule:
A-shares reach the China vendor first and fall back when it fails; statements
are served by announcement date; prices, announcements, forum posts and macro
series never reach past the analysis date.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.symbols import a_share_parts, is_china_a_share
from tradingagents.dataflows.vendors.china import fundamentals, macro, market, news, social

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_ak(monkeypatch):
    """A stub AKShare installed in every China module, with the day cache bypassed."""
    stub = SimpleNamespace()
    for module in (market, fundamentals, news, social, macro):
        monkeypatch.setattr(module, "akshare", lambda: stub)
    for module in (market, fundamentals, news):
        monkeypatch.setattr(module, "cached_frame", lambda kind, key, fetch: fetch())
    return stub


# --- symbols ------------------------------------------------------------------

def test_a_share_parts_reads_every_exchange_spelling():
    assert a_share_parts("600519.SS") == ("600519", "SH")
    assert a_share_parts("600519.sh") == ("600519", "SH")
    assert a_share_parts("300010.SZ") == ("300010", "SZ")
    assert a_share_parts("830799.BJ") == ("830799", "BJ")
    for other in ("AAPL", "0700.HK", "60051.SS", "BTC-USD", None):
        assert not is_china_a_share(other)


# --- routing ------------------------------------------------------------------

def _record(name, calls, result=None, raises=None):
    def impl(*args, **kwargs):
        calls.append(name)
        if raises:
            raise raises
        return result or f"{name} answered"
    return impl


def test_a_share_goes_to_the_china_vendor_first(monkeypatch):
    calls = []
    monkeypatch.setitem(router.VENDOR_METHODS, "get_news", {
        "yfinance": _record("yfinance", calls), "china": _record("china", calls)})
    assert router.route_to_vendor("get_news", "300010.SZ", "2026-04-01", "2026-04-08") == "china answered"
    assert calls == ["china"]


def test_other_tickers_never_reach_the_china_vendor(monkeypatch):
    calls = []
    monkeypatch.setitem(router.VENDOR_METHODS, "get_news", {
        "yfinance": _record("yfinance", calls), "china": _record("china", calls)})
    router.route_to_vendor("get_news", "AAPL", "2026-04-01", "2026-04-08")
    assert calls == ["yfinance"]


def test_a_failed_china_feed_falls_back_to_the_configured_chain(monkeypatch):
    calls = []
    monkeypatch.setitem(router.VENDOR_METHODS, "get_news", {
        "yfinance": _record("yfinance", calls),
        "china": _record("china", calls, raises=VendorUnavailableError("blocked"))})
    assert router.route_to_vendor("get_news", "300010.SZ", "2026-04-01", "2026-04-08") == "yfinance answered"
    assert calls == ["china", "yfinance"]


def test_china_auto_vendor_off_sends_a_shares_through_the_configured_chain(monkeypatch):
    calls = []
    set_config({"china_auto_vendor": False})
    monkeypatch.setitem(router.VENDOR_METHODS, "get_news", {
        "yfinance": _record("yfinance", calls), "china": _record("china", calls)})
    router.route_to_vendor("get_news", "300010.SZ", "2026-04-01", "2026-04-08")
    assert calls == ["yfinance"]


def test_china_macro_aliases_route_to_china_and_others_to_fred(monkeypatch):
    calls = []
    monkeypatch.setitem(router.VENDOR_METHODS, "get_macro_indicators", {
        "fred": _record("fred", calls), "china": _record("china", calls)})
    router.route_to_vendor("get_macro_indicators", "cn_cpi", "2026-04-15", None)
    router.route_to_vendor("get_macro_indicators", "cpi", "2026-04-15", None)
    assert calls == ["china", "fred"]


def test_route_symbol_routes_a_method_without_a_ticker(monkeypatch):
    calls = []
    monkeypatch.setitem(router.VENDOR_METHODS, "get_global_news", {
        "yfinance": _record("yfinance", calls), "china": _record("china", calls)})
    router.route_to_vendor("get_global_news", "2026-04-15", 7, 10, route_symbol="600519.SS")
    router.route_to_vendor("get_global_news", "2026-04-15", 7, 10)
    assert calls == ["china", "yfinance"]


# --- prices ---------------------------------------------------------------------

def _bars(dates):
    return pd.DataFrame({
        "日期": dates, "开盘": 10.0, "收盘": 10.5, "最高": 11.0, "最低": 9.5,
        "成交量": 1000, "成交额": 1.05e6, "换手率": 1.2, "涨跌幅": 0.5,
    })


def test_prices_stop_at_the_analysis_date_and_volume_is_in_shares(fake_ak):
    fake_ak.stock_zh_a_hist = lambda **kw: _bars(["2026-04-13", "2026-04-14", "2026-04-15", "2026-04-16"])
    frame = market.load_ohlcv_cn("300010.SZ", "2026-04-15")
    assert frame["Date"].max() == pd.Timestamp("2026-04-15")
    assert frame["Volume"].iloc[-1] == 100_000  # 1000 lots of 100 shares


def test_stock_data_names_the_board_and_its_price_limit(fake_ak):
    fake_ak.stock_zh_a_hist = lambda **kw: _bars(["2026-04-14", "2026-04-15"])
    out = market.get_stock_data_cn("300010.SZ", "2026-04-14", "2026-04-15")
    assert "ChiNext" in out and "±20%" in out and "前复权" in out


# --- statements -----------------------------------------------------------------

def _profit_sheet():
    return pd.DataFrame({
        "REPORT_DATE": ["2026-03-31", "2025-12-31", "2025-09-30"],
        "REPORT_DATE_NAME": ["2026一季报", "2025年报", "2025三季报"],
        "NOTICE_DATE": ["2026-04-30", "2026-04-30", "2025-10-27"],
        "UPDATE_DATE": ["2026-04-30", "2026-08-28", "2025-10-27"],
        "TOTAL_OPERATE_INCOME": [2e8, 10e8, 7.7e8],
        "TOTAL_OPERATE_INCOME_YOY": [4.5, 33.0, 39.4],
        "PARENT_NETPROFIT": [-0.5e8, 0.7e8, 0.9e8],
        "PARENT_NETPROFIT_YOY": [-250.0, 12.0, -17.0],
        "DEDUCT_PARENT_NETPROFIT": [-0.6e8, -1.3e8, 0.4e8],
        "BASIC_EPS": [-0.03, 0.03, 0.04],
    })


def test_a_report_is_served_only_once_announced(fake_ak):
    fake_ak.stock_profit_sheet_by_report_em = lambda symbol: _profit_sheet()
    before = fundamentals.get_income_statement_cn("300010.SZ", "quarterly", "2026-04-15")
    assert "2025三季报" in before
    assert "2025年报" not in before and "2026一季报" not in before
    after = fundamentals.get_income_statement_cn("300010.SZ", "quarterly", "2026-05-06")
    assert "2025年报" in after and "2026一季报" in after


def test_income_figures_are_flagged_year_to_date_and_later_revisions_noted(fake_ak):
    fake_ak.stock_profit_sheet_by_report_em = lambda symbol: _profit_sheet()
    out = fundamentals.get_income_statement_cn("300010.SZ", "quarterly", "2026-05-06")
    assert "YEAR-TO-DATE" in out
    assert "2025年报 was revised on 2026-08-28" in out


def test_no_report_announced_yet_is_no_data(fake_ak):
    fake_ak.stock_profit_sheet_by_report_em = lambda symbol: _profit_sheet()
    with pytest.raises(NoMarketDataError):
        fundamentals.get_income_statement_cn("300010.SZ", "quarterly", "2025-10-01")


def test_a_past_run_withholds_the_live_profile(fake_ak, monkeypatch):
    fake_ak.stock_value_em = lambda symbol: pd.DataFrame({
        "数据日期": ["2026-04-14", "2026-04-15", "2026-04-16"], "当日收盘价": [6.1, 6.08, 9.9],
        "总市值": [1.2e10] * 3, "流通市值": [1e10] * 3, "总股本": [2e9] * 3,
        "PE(TTM)": [100.0] * 3, "PE(静)": [170.0] * 3, "市净率": [7.0] * 3,
        "PEG值": [1.0] * 3, "市现率": [-50.0] * 3, "市销率": [12.0] * 3})
    fake_ak.stock_profit_sheet_by_report_em = lambda symbol: _profit_sheet()
    fake_ak.stock_individual_info_em = lambda symbol: pytest.fail("live profile fetched for a past date")
    monkeypatch.setattr(fundamentals, "fetch_announcements",
                        lambda t, s, e: pd.DataFrame(columns=["time", "title", "link"]))
    out = fundamentals.get_fundamentals_cn("300010.SZ", "2026-04-15")
    assert "Close: 6.08" in out and "9.90" not in out
    assert "withheld for a past date" in out


def test_risk_warning_status_reads_the_short_name():
    assert fundamentals.risk_warning_status("*ST广糖").startswith("*ST")
    assert fundamentals.risk_warning_status("ST豆神").startswith("ST")
    assert fundamentals.risk_warning_status("贵州茅台") is None


# --- news and announcements -------------------------------------------------------

def test_announcements_are_trimmed_to_the_window(fake_ak):
    fake_ak.stock_zh_a_disclosure_report_cninfo = lambda **kw: pd.DataFrame({
        "公告标题": ["inside", "after the date"],
        "公告时间": ["2026-04-15 17:18:00", "2026-04-16 09:00:00"],
        "公告链接": ["", ""]})
    frame = news.fetch_announcements("300010.SZ", "2026-04-01", "2026-04-15")
    assert frame["title"].tolist() == ["inside"]


def test_a_past_window_the_news_feed_cannot_reach_is_unavailable_not_empty(fake_ak, monkeypatch):
    fake_ak.stock_news_em = lambda symbol: pd.DataFrame({
        "新闻标题": ["LATER HEADLINE"], "新闻内容": ["x"], "发布时间": ["2026-10-12 19:06:00"],
        "文章来源": ["财联社"], "新闻链接": [""]})
    monkeypatch.setattr(news, "fetch_announcements",
                        lambda t, s, e: pd.DataFrame(columns=["time", "title", "link"]))
    out = news.get_news_cn("300010.SZ", "2026-04-08", "2026-04-15")
    assert "unavailable for 2026-04-08..2026-04-15" in out
    assert "LATER HEADLINE" not in out


# --- forum ------------------------------------------------------------------------

def _forum_page(posts):
    return "<html><script>var article_list=" + json.dumps({"re": posts}) + ";var other=1;</script>"


def test_forum_posts_are_trimmed_to_the_window_and_pinned_posts_skipped(monkeypatch):
    posts = [
        {"post_title": "pinned notice", "post_publish_time": "2026-10-15 10:00:00", "post_top_status": 1},
        {"post_title": "after the date", "post_publish_time": "2026-10-16 10:00:00"},
        {"post_title": "in the window", "post_publish_time": "2026-10-15 09:00:00",
         "post_click_count": 50, "post_comment_count": 3},
        {"post_title": "older", "post_publish_time": "2026-10-01 09:00:00"},
    ]
    response = SimpleNamespace(text=_forum_page(posts), raise_for_status=lambda: None)
    monkeypatch.setattr(social.requests, "get", lambda *a, **k: response)
    out = social.fetch_guba_posts("300010.SZ", "2026-10-09", "2026-10-15", pages=1)
    assert "in the window" in out and "50 views" in out
    assert "after the date" not in out and "pinned notice" not in out and "older" not in out


def test_popularity_rank_stops_at_the_analysis_date(fake_ak):
    fake_ak.stock_hot_rank_detail_em = lambda symbol: pd.DataFrame({
        "时间": ["2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16"], "排名": [900, 700, 500, 1]})
    out = social.fetch_popularity_rank("300010.SZ", "2026-10-15")
    assert "rank 500" in out and "rank 1\n" not in out and not out.endswith("rank 1")


# --- macro ------------------------------------------------------------------------

def test_a_monthly_figure_waits_for_its_publication_lag(fake_ak):
    fake_ak.macro_china_cpi = lambda: pd.DataFrame({
        "月份": ["2026年03月份", "2026年02月份"], "全国-同比增长": [1.0, 1.3], "全国-环比增长": [-0.7, 1.0]})
    early = macro.get_macro_data_cn("cn_cpi", "2026-04-10")
    assert "Latest period: 2026年02月份" in early
    later = macro.get_macro_data_cn("cn_cpi", "2026-04-16")
    assert "Latest period: 2026年03月份" in later


def test_an_unknown_alias_is_left_for_fred():
    with pytest.raises(NoMarketDataError):
        macro.get_macro_data_cn("fed_funds_rate", "2026-04-15")


# --- agent context ----------------------------------------------------------------

def test_instrument_context_names_the_risk_warning_on_a_live_run():
    from tradingagents.agents.context import build_instrument_context

    identity = {"company_name": "Doushen", "local_name": "ST豆神",
                "risk_warning": "ST — other risk warning (其他风险警示)", "local_industry": "教育"}
    live = build_instrument_context("300010.SZ", "stock", identity, None)
    assert "ST豆神" in live and "Risk warning" in live
    past = build_instrument_context("300010.SZ", "stock", identity, "2020-01-02")
    assert "ST豆神" not in past and "Risk warning" not in past


def test_analyst_prompts_carry_a_share_guidance_only_for_a_shares():
    from tradingagents.agents.context import china_a_share_note

    assert "YEAR-TO-DATE" in china_a_share_note("300010.SZ", "fundamentals")
    assert "CNINFO" in china_a_share_note("600519.SS", "news")
    assert china_a_share_note("AAPL", "fundamentals") == ""


def test_the_a_share_sentiment_prompt_reads_the_chinese_sources():
    from tradingagents.agents.analysts.sentiment_analyst import _build_a_share_system_message

    message = _build_a_share_system_message(
        ticker="300010.SZ", start_date="2026-10-08", end_date="2026-10-15",
        news_block="NEWS", forum_block="FORUM", rank_block="RANK")
    assert "股吧" in message and "FORUM" in message and "RANK" in message
    assert "StockTwits" not in message and "Reddit" not in message


# --- report header ----------------------------------------------------------------

def test_report_header_names_the_china_vendor_an_a_share_was_served_from():
    from tradingagents.reporting import vendors_used

    settings = {"data_vendors": {"core_stock_apis": "yfinance", "fundamental_data": "sec_edgar,yfinance",
                                 "prediction_markets": "polymarket"},
                "tool_vendors": {}, "china_auto_vendor": True}
    a_share = vendors_used(settings, "300010.SZ")
    assert a_share["core_stock_apis"] == "china,yfinance"
    assert a_share["fundamental_data"] == "china,sec_edgar,yfinance"
    assert a_share["prediction_markets"] == "polymarket"
    assert vendors_used(settings, "AAPL")["core_stock_apis"] == "yfinance"
    assert vendors_used({**settings, "china_auto_vendor": False}, "300010.SZ")["core_stock_apis"] == "yfinance"
