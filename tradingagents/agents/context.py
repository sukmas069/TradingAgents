"""Prompt context shared by the agents: instrument identity, output language and portfolio."""

import functools
import logging
from collections.abc import Mapping
from typing import Any

from tradingagents.dataflows.date_window import is_historical
from tradingagents.dataflows.vendors.yahoo.fundamentals import get_company_profile

logger = logging.getLogger(__name__)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    # The labelled lines are read by the program, so they keep their English
    # label and value: a translated rating line leaves the reader prose to
    # search, where a negated rating ("not a Sell") reads as the call (#1435).
    return (
        f" Write your entire response in {lang}, except the labelled lines the format"
        f" asks for (the \"**Rating**:\" line, \"FINAL TRANSACTION PROPOSAL:\"):"
        f" keep their label and value in English, exactly as specified."
    )


def opponent_argument_or_opening(text: str, opponent: str) -> str:
    """Opponent's latest argument, or an explicit opening marker when empty.

    The first speaker in each debate round receives an empty opponent response;
    interpolating it into a "refute the opponent" prompt makes the model
    fabricate the other side's position. Returning a clear "has not spoken yet"
    marker instead lets it open with its own case (#1176).
    """
    text = (text or "").strip()
    if text:
        return text
    return f"(The {opponent} has not spoken yet — open the debate with your own case.)"


def _clean_identity_value(value: Any) -> str | None:
    """Return a trimmed string, or None for empty / placeholder-ish values."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "nan", "null"}:
        return None
    return cleaned


def resolve_instrument_identity(ticker: str) -> dict:
    """Resolve deterministic identity metadata (company name, sector, …) for a ticker.

    This exists to stop the pipeline from hallucinating a *different* company
    when a chart pattern suggests a different industry than the real one
    (#814): without a ground-truth name, the market analyst would pattern-match
    the price action to a narrative and invent an identity that then cascaded
    through every downstream agent.

    Best-effort by design: if yfinance is unavailable, rate-limited, or doesn't
    recognise the ticker, we return ``{}`` and the caller falls back to
    ticker-only context rather than failing before analysis starts. An answer
    is cached for the process; a failed lookup is asked again next time.

    Identity resolves for the same instrument the price path fetches
    (``XAUUSD`` -> ``GC=F``, #983).
    """
    try:
        return _identity(ticker)
    except Exception as exc:  # noqa: BLE001 — fail open, never block the run
        logger.debug("Could not resolve instrument identity for %s: %s", ticker, exc)
        return {}


@functools.lru_cache(maxsize=256)
def _identity(ticker: str) -> dict:
    """The vendor's identity fields for ``ticker``; raises if the lookup fails."""
    info = get_company_profile(ticker)
    identity: dict[str, str] = {}
    company_name = _clean_identity_value(info.get("longName")) or _clean_identity_value(
        info.get("shortName")
    )
    if company_name:
        identity["company_name"] = company_name
    for source_key, target_key in (
        ("sector", "sector"),
        ("industry", "industry"),
        ("exchange", "exchange"),
        ("quoteType", "quote_type"),
    ):
        value = _clean_identity_value(info.get(source_key))
        if value:
            identity[target_key] = value
    identity.update(_china_identity(ticker))
    return identity


def _china_identity(ticker: str) -> dict[str, str]:
    """An A-share's Chinese short name and risk-warning status, from East Money.

    The short name carries the ST / *ST prefix, which Yahoo's English name does
    not, and a risk warning changes how every agent should read the stock.
    Best-effort: an unreachable East Money leaves the Yahoo identity alone.
    """
    from tradingagents.dataflows.router import china_auto_enabled
    from tradingagents.dataflows.symbols import is_china_a_share
    from tradingagents.dataflows.vendors.china.fundamentals import (
        get_profile_cn,
        risk_warning_status,
    )

    if not (china_auto_enabled() and is_china_a_share(ticker)):
        return {}
    try:
        profile = get_profile_cn(ticker)
    except Exception as exc:  # noqa: BLE001 — identity is best-effort
        logger.debug("No East Money profile for %s: %s", ticker, exc)
        return {}
    out: dict[str, str] = {}
    short_name = _clean_identity_value(str(profile.get("股票简称") or ""))
    if short_name:
        out["local_name"] = short_name
        status = risk_warning_status(short_name)
        if status:
            out["risk_warning"] = status
    industry = _clean_identity_value(str(profile.get("行业") or ""))
    if industry:
        out["local_industry"] = industry
    return out


def build_instrument_context(
    ticker: str,
    asset_type: str = "stock",
    identity: Mapping[str, str] | None = None,
    trade_date: str | None = None,
) -> str:
    """Describe the exact instrument so agents preserve identity and ticker.

    When ``identity`` is provided (resolved deterministically via
    :func:`resolve_instrument_identity`), the company name and business
    classification are injected so agents anchor to the real company rather
    than pattern-matching the price chart to a wrong one (#814).

    That profile carries no historical vintage: it describes the company today.
    A run dated earlier gets the current name alone, as a way to tell the
    company apart from others rather than as what it was called then; a sector,
    industry or exchange it holds today is not given, since it may not have held
    on the analysis date.
    """
    is_crypto = asset_type == "crypto"
    instrument_label = "asset" if is_crypto else "instrument"
    context = (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "The tools serve this instrument; refer to it by this exact ticker in every report and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
    )

    identity = identity or {}
    name = identity.get("company_name") or identity.get("name")
    label = "Name" if is_crypto else "Company"
    details = []
    if is_historical(trade_date):
        if name:
            details.append(
                f"{label}: {name} (its current name, given only to identify it; "
                f"on {trade_date} it may have been named differently)"
            )
    else:
        if name:
            details.append(f"{label}: {name}")
        sector, industry = identity.get("sector"), identity.get("industry")
        if sector and industry:
            details.append(f"Business classification: {sector} / {industry}")
        elif sector:
            details.append(f"Sector: {sector}")
        elif industry:
            details.append(f"Industry: {industry}")
        if identity.get("exchange"):
            details.append(f"Exchange: {identity['exchange']}")
        if identity.get("local_name"):
            local = f"Chinese short name: {identity['local_name']}"
            if identity.get("local_industry"):
                local += f" (industry 行业: {identity['local_industry']})"
            details.append(local)
        if identity.get("risk_warning"):
            details.append(
                f"Risk warning: {identity['risk_warning']}; the exchange has flagged this stock, "
                "so weigh delisting and disclosure risk in every recommendation"
            )

    if details:
        context += (
            f" Resolved identity: {'; '.join(details)}. "
            "Do not substitute a different company or ticker unless a tool "
            "result explicitly disproves this resolved identity."
        )

    if is_crypto:
        context += (
            " Treat it as a crypto asset rather than a company, and do not "
            "assume company fundamentals are available."
        )
    return context


_CHINA_A_SHARE_NOTES = {
    "market": (
        " This is a China A-share: prices are forward-adjusted (前复权) CNY daily bars from East Money."
        " The get_stock_data header gives the board's daily price limit; a close at the limit"
        " (涨停/跌停) is a forced stop, not a settled price, and distorts momentum readings."
        " Turnover % (换手率) is a key measure of A-share trading activity, and trading settles T+1."
    ),
    "news": (
        " This is a China A-share. get_news returns East Money company news and the company's"
        " CNINFO exchange announcements: treat the announcements (results, risk warnings,"
        " regulatory letters, major contracts, shareholding changes) as primary disclosures."
        " get_global_news returns China's domestic headlines (CCTV evening news, Cailian Press)"
        " followed by international headlines. For macro, use the China series first:"
        " get_macro_indicators with 'cn_cpi', 'cn_ppi', 'cn_pmi', 'cn_m2', 'cn_lpr', 'cn_gdp',"
        " 'cn_shibor'; US series still matter for global risk appetite."
    ),
    "fundamentals": (
        " This is a China A-share. Statements come from East Money as announced. Income and"
        " cash-flow figures are YEAR-TO-DATE (Q3 report = nine months): derive a single quarter"
        " by subtracting the previous report of the same year before comparing quarters."
        " get_fundamentals gives the daily valuation, the risk-warning (ST/*ST) status and the"
        " past year's risk-related announcements; state what they imply (delisting risk,"
        " regulatory investigation, audit opinion, results warnings). get_insider_transactions"
        " lists shareholding-change announcements (增持/减持/回购/解禁)."
    ),
}


def china_a_share_note(ticker: str, role: str) -> str:
    """Prompt guidance for an analyst when the instrument is a China A-share, else ''."""
    from tradingagents.dataflows.router import china_auto_enabled
    from tradingagents.dataflows.symbols import is_china_a_share

    if not (china_auto_enabled() and is_china_a_share(ticker)):
        return ""
    return _CHINA_A_SHARE_NOTES.get(role, "")


def get_instrument_context_from_state(state: Mapping[str, Any]) -> str:
    """Return the instrument context for the current run.

    Prefers the identity-resolved context computed once at run start and
    stored on the state (see ``TradingAgentsGraph.resolve_instrument_context``).
    Falls back to a ticker-only context — with no network lookup — when the
    state was constructed without it (bare programmatic states, tests), so a
    consumer is never forced to make a yfinance call mid-graph.
    """
    context = state.get("instrument_context")
    if isinstance(context, str) and context.strip():
        return context
    return build_instrument_context(
        str(state["company_of_interest"]),
        state.get("asset_type", "stock"),
    )


def report_or_absent(text: str, source: str) -> str:
    """An analyst's report, or a marker saying it was never produced.

    A report is empty when its analyst was not selected, refused, or returned
    nothing. Interpolating that into a labelled section presents an absence as a
    blank finding, and the reading agent fills it in from nothing, the same way
    an empty opponent argument used to invite an invented rebuttal (#1176).
    """
    text = (text or "").strip()
    if text:
        return text
    return f"(No {source} report in this run: it is not available, not an empty finding.)"


def get_portfolio_context_from_state(state: Mapping[str, Any]) -> str:
    """Return the caller's portfolio block, or a notice that none was given.

    A run without portfolio context must not read as a flat book: the agents
    would otherwise size as if the caller held nothing, which is a claim about
    an account we were never told about.
    """
    context = state.get("portfolio_context")
    if isinstance(context, str) and context.strip():
        return context
    return (
        "Portfolio context: not provided. You do not know the caller's current "
        "holdings or cash, so do not assume a flat book; give direction and "
        "sizing guidance in terms the caller can apply to their own position."
    )
