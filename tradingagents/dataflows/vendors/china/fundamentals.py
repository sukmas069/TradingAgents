"""A-share statements as announced, daily valuation, profile and risk-warning status.

East Money publishes every report with the date it was announced
(``NOTICE_DATE``), so a run dated ``as_of_date`` serves only the reports that
were public by then, the way SEC EDGAR serves US filers. A report revised
afterwards keeps East Money's current figures; its later ``UPDATE_DATE`` is
shown so the reader knows the numbers may differ from those first announced.

Income and cash-flow figures in A-share reports are year-to-date: the Q3
report covers nine months. The output says so, since reading them as single
quarters overstates every quarter after the first.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from tradingagents.dataflows.date_window import is_historical
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.vendors.china.common import (
    CNINFO,
    EAST_MONEY,
    ak_call,
    akshare,
    cached_frame,
    require_a_share,
    table,
)
from tradingagents.dataflows.vendors.china.news import (
    RISK_KEYWORDS,
    fetch_announcements,
    format_announcements,
)

# Line items per statement: East Money field, label. Money fields are shown in
# 100 million CNY (亿元); per-share and percentage fields as reported.
_STATEMENTS = {
    "balance_sheet": ("Balance Sheet", "stock_balance_sheet_by_report_em", "bs", [
        ("TOTAL_ASSETS", "Total assets 资产总计"),
        ("TOTAL_CURRENT_ASSETS", "Current assets 流动资产"),
        ("MONETARYFUNDS", "Cash 货币资金"),
        ("TRADE_FINASSET_NOTFVTPL", "Trading financial assets 交易性金融资产"),
        ("ACCOUNTS_RECE", "Accounts receivable 应收账款"),
        ("INVENTORY", "Inventory 存货"),
        ("GOODWILL", "Goodwill 商誉"),
        ("INTANGIBLE_ASSET", "Intangible assets 无形资产"),
        ("TOTAL_LIABILITIES", "Total liabilities 负债合计"),
        ("TOTAL_CURRENT_LIAB", "Current liabilities 流动负债"),
        ("SHORT_LOAN", "Short-term borrowings 短期借款"),
        ("LONG_LOAN", "Long-term borrowings 长期借款"),
        ("BOND_PAYABLE", "Bonds payable 应付债券"),
        ("TOTAL_PARENT_EQUITY", "Equity to parent 归母权益"),
        ("TOTAL_EQUITY", "Total equity 股东权益合计"),
    ]),
    "income_statement": ("Income Statement", "stock_profit_sheet_by_report_em", "pl", [
        ("TOTAL_OPERATE_INCOME", "Revenue 营业总收入"),
        ("TOTAL_OPERATE_INCOME_YOY", "Revenue YoY % 营收同比"),
        ("OPERATE_COST", "Cost of revenue 营业成本"),
        ("SALE_EXPENSE", "Selling expense 销售费用"),
        ("MANAGE_EXPENSE", "Admin expense 管理费用"),
        ("RESEARCH_EXPENSE", "R&D expense 研发费用"),
        ("FINANCE_EXPENSE", "Finance expense 财务费用"),
        ("INVEST_INCOME", "Investment income 投资收益"),
        ("FAIRVALUE_CHANGE_INCOME", "Fair-value gains 公允价值变动收益"),
        ("ASSET_IMPAIRMENT_INCOME", "Asset impairment 资产减值损失"),
        ("CREDIT_IMPAIRMENT_INCOME", "Credit impairment 信用减值损失"),
        ("OPERATE_PROFIT", "Operating profit 营业利润"),
        ("TOTAL_PROFIT", "Pre-tax profit 利润总额"),
        ("NETPROFIT", "Net profit 净利润"),
        ("PARENT_NETPROFIT", "Net profit to parent 归母净利润"),
        ("PARENT_NETPROFIT_YOY", "Net profit to parent YoY % 归母净利同比"),
        ("DEDUCT_PARENT_NETPROFIT", "Recurring net profit 扣非归母净利润"),
        ("BASIC_EPS", "Basic EPS (CNY) 基本每股收益"),
    ]),
    "cashflow": ("Cash Flow", "stock_cash_flow_sheet_by_report_em", "cf", [
        ("SALES_SERVICES", "Cash from sales 销售商品收到的现金"),
        ("NETCASH_OPERATE", "Operating cash flow 经营活动现金流净额"),
        ("CONSTRUCT_LONG_ASSET", "Capex 购建长期资产支付的现金"),
        ("NETCASH_INVEST", "Investing cash flow 投资活动现金流净额"),
        ("NETCASH_FINANCE", "Financing cash flow 筹资活动现金流净额"),
        ("CCE_ADD", "Net change in cash 现金净增加额"),
        ("END_CCE", "Cash at period end 期末现金余额"),
    ]),
}

# Fields shown as reported rather than scaled to 100 million CNY.
_UNSCALED = {"TOTAL_OPERATE_INCOME_YOY", "PARENT_NETPROFIT_YOY", "BASIC_EPS"}
_YI = 1e8


def _east_money_symbol(code: str, exchange: str) -> str:
    return f"{exchange}{code}"


def _statement_frame(ticker: str, which: str) -> pd.DataFrame:
    code, exchange = require_a_share(ticker)
    _title, func_name, kind, _items = _STATEMENTS[which]
    symbol = _east_money_symbol(code, exchange)
    ak = akshare()
    return cached_frame(kind, symbol, lambda: ak_call(
        getattr(ak, func_name), symbol=symbol, what=f"{EAST_MONEY} {which.replace('_', ' ')} for {symbol}",
    ))


def _announced_by(frame: pd.DataFrame, as_of_date: str | None) -> pd.DataFrame:
    """Reports announced on or before ``as_of_date`` (all of them for a live run)."""
    frame = frame.copy()
    frame["NOTICE_DATE"] = pd.to_datetime(frame["NOTICE_DATE"], errors="coerce")
    frame["UPDATE_DATE"] = pd.to_datetime(frame.get("UPDATE_DATE"), errors="coerce")
    frame["REPORT_DATE"] = pd.to_datetime(frame["REPORT_DATE"], errors="coerce")
    if as_of_date:
        frame = frame[frame["NOTICE_DATE"].notna() & (frame["NOTICE_DATE"] <= pd.to_datetime(as_of_date))]
    return frame.sort_values("REPORT_DATE", ascending=False)


def _statement(ticker: str, freq: str, as_of_date: str | None, which: str) -> str:
    title, _func, _kind, items = _STATEMENTS[which]
    code, _ = require_a_share(ticker)
    frame = _announced_by(_statement_frame(ticker, which), as_of_date)
    if freq.lower() == "annual":
        frame = frame[frame["REPORT_DATE"].dt.month == 12]
    frame = frame.head(8 if freq.lower() == "annual" else 10)
    if frame.empty:
        raise NoMarketDataError(ticker, code, f"no {title.lower()} announced by {as_of_date or 'today'}")

    rows = []
    for _, report in frame.iterrows():
        row = {
            "Period": report.get("REPORT_DATE_NAME") or f"{report['REPORT_DATE']:%Y-%m-%d}",
            "Announced": f"{report['NOTICE_DATE']:%Y-%m-%d}",
        }
        for field, label in items:
            value = pd.to_numeric(report.get(field), errors="coerce")
            if pd.isna(value):
                row[label] = ""
            elif field in _UNSCALED:
                row[label] = round(float(value), 2)
            else:
                row[label] = round(float(value) / _YI, 3)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("Period").T.reset_index().rename(columns={"index": "Item"})

    notes = [
        f"# {title} for {ticker.upper()} ({freq}), as announced by {as_of_date or 'today'}",
        f"# Source: {EAST_MONEY}. Money in 100 million CNY (亿元); YoY in %; EPS in CNY.",
    ]
    if which != "balance_sheet":
        notes.append("# Figures are YEAR-TO-DATE (累计): 一季报 = Q1, 中报 = H1, 三季报 = 9 months, 年报 = full year. "
                     "A single quarter is the difference between consecutive reports of the same year.")
    if as_of_date:
        revised = frame[frame["UPDATE_DATE"].notna() & (frame["UPDATE_DATE"] > pd.to_datetime(as_of_date))]
        for _, report in revised.iterrows():
            notes.append(f"# Note: {report.get('REPORT_DATE_NAME')} was revised on {report['UPDATE_DATE']:%Y-%m-%d}, "
                         f"after {as_of_date}; the figures shown may differ from those first announced.")
    return "\n".join(notes) + "\n\n" + table(out)


def get_balance_sheet_cn(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    return _statement(ticker, freq, as_of_date, "balance_sheet")


def get_income_statement_cn(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    return _statement(ticker, freq, as_of_date, "income_statement")


def get_cashflow_cn(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    return _statement(ticker, freq, as_of_date, "cashflow")


def _valuation(code: str, as_of_date: str | None) -> list[str]:
    """Market cap and multiples on the last trading day on or before ``as_of_date``."""
    ak = akshare()
    frame = cached_frame("value", code, lambda: ak_call(
        ak.stock_value_em, symbol=code, what=f"{EAST_MONEY} valuation history for {code}",
    ))
    frame = frame.copy()
    frame["数据日期"] = pd.to_datetime(frame["数据日期"], errors="coerce")
    if as_of_date:
        frame = frame[frame["数据日期"] <= pd.to_datetime(as_of_date)]
    if frame.empty:
        return []
    row = frame.sort_values("数据日期").iloc[-1]

    def num(key, scale=1.0, digits=2):
        value = pd.to_numeric(row.get(key), errors="coerce")
        return "N/A" if pd.isna(value) else f"{value / scale:,.{digits}f}"

    return [
        f"Valuation date: {row['数据日期']:%Y-%m-%d}",
        f"Close: {num('当日收盘价')} CNY",
        f"Market cap 总市值: {num('总市值', _YI)} 亿元 (100M CNY)",
        f"Free-float market cap 流通市值: {num('流通市值', _YI)} 亿元",
        f"Total shares 总股本: {num('总股本', 1e4, 0)} 万股 (10k shares)",
        f"PE (TTM) 市盈率TTM: {num('PE(TTM)')} (negative = loss-making)",
        f"PE (static, last annual) 静态市盈率: {num('PE(静)')}",
        f"Price to book 市净率: {num('市净率')}",
        f"PEG: {num('PEG值')}",
        f"Price to operating cash flow 市现率: {num('市现率')}",
        f"Price to sales 市销率: {num('市销率')}",
    ]


def get_profile_cn(ticker: str) -> dict:
    """East Money's current profile: short name (with any ST prefix), industry, listing date."""
    code, _ = require_a_share(ticker)
    ak = akshare()
    frame = ak_call(ak.stock_individual_info_em, symbol=code, what=f"{EAST_MONEY} profile for {code}")
    if frame is None or frame.empty:
        return {}
    return {str(k): v for k, v in zip(frame["item"], frame["value"], strict=True)}


def risk_warning_status(short_name: str | None) -> str | None:
    """The risk-warning label carried by an A-share short name, if any."""
    if not short_name:
        return None
    name = str(short_name).strip().upper()
    if name.startswith("*ST"):
        return "*ST — delisting risk warning (退市风险警示)"
    if name.startswith("ST"):
        return "ST — other risk warning (其他风险警示)"
    return None


def _yi(report: pd.Series, field: str) -> str:
    value = pd.to_numeric(report.get(field), errors="coerce")
    return "N/A" if pd.isna(value) else f"{value / _YI:,.3f} 亿元"


def _pct(report: pd.Series, field: str) -> str:
    value = pd.to_numeric(report.get(field), errors="coerce")
    return "N/A" if pd.isna(value) else f"{value:+.2f}%"


def get_fundamentals_cn(ticker: str, as_of_date: str | None = None) -> str:
    """Valuation, profile, latest results and risk-related announcements for an A-share."""
    code, exchange = require_a_share(ticker)
    historical = is_historical(as_of_date)
    lines = [f"# Company Fundamentals for {ticker.upper()}", f"# Point-in-time as of: {as_of_date or 'today'}", ""]

    lines.append(f"## Valuation ({EAST_MONEY}, daily history)")
    try:
        lines += _valuation(code, as_of_date) or ["<no valuation history on or before this date>"]
    except VendorUnavailableError as exc:
        lines.append(f"<unavailable: {exc}>")

    lines += ["", "## Profile"]
    if historical:
        lines.append("Profile fields (short name, industry, ST status) are withheld for a past date: "
                     "East Money serves only today's values, and a stock's name and risk-warning status "
                     "change over time. Check the risk announcements below for its status on that date.")
    else:
        try:
            profile = get_profile_cn(ticker)
            for key in ("股票简称", "行业", "上市时间", "总股本", "流通股"):
                if key in profile:
                    lines.append(f"{key}: {profile[key]}")
            status = risk_warning_status(profile.get("股票简称"))
            lines.append(f"Risk-warning status 风险警示: {status or 'none (not ST)'}")
        except VendorUnavailableError as exc:
            lines.append(f"<unavailable: {exc}>")

    lines += ["", "## Latest announced results (year-to-date figures)"]
    try:
        frame = _announced_by(_statement_frame(ticker, "income_statement"), as_of_date)
        if frame.empty:
            lines.append("<no report announced by this date>")
        for _, report in frame.head(2).iterrows():
            lines.append(
                f"- {report.get('REPORT_DATE_NAME')} (announced {report['NOTICE_DATE']:%Y-%m-%d}): "
                f"revenue {_yi(report, 'TOTAL_OPERATE_INCOME')} ({_pct(report, 'TOTAL_OPERATE_INCOME_YOY')} YoY); "
                f"net profit to parent {_yi(report, 'PARENT_NETPROFIT')} "
                f"({_pct(report, 'PARENT_NETPROFIT_YOY')} YoY); "
                f"recurring net profit {_yi(report, 'DEDUCT_PARENT_NETPROFIT')}"
            )
    except VendorUnavailableError as exc:
        lines.append(f"<unavailable: {exc}>")

    end = as_of_date or datetime.now().strftime("%Y-%m-%d")
    start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
    lines += ["", f"## Risk-related announcements ({CNINFO}, {start} to {end})",
              "Risk warnings, delisting, regulatory letters and penalties, results warnings, "
              "impairments, litigation, pledges and guarantees:"]
    try:
        listed = format_announcements(fetch_announcements(ticker, start, end), limit=40, keywords=RISK_KEYWORDS)
        lines.append(listed or "<none in this window>")
    except VendorUnavailableError as exc:
        lines.append(f"<unavailable: {exc}>")

    return "\n".join(lines)
