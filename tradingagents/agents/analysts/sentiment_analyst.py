"""Sentiment analyst: one sentiment report from three sources.

The node fetches its sources before calling the model and puts them in the
prompt, so the model reports on data it was given rather than inventing posts:

  1. News headlines: Yahoo Finance
  2. StockTwits messages: the cashtag stream, with Bullish/Bearish tags
  3. Reddit posts: r/wallstreetbets, r/stocks, r/investing, or crypto communities for a crypto pair

A China A-share reads Chinese sources instead, since the US boards barely
discuss it: East Money news with CNINFO announcements, East Money stock-forum
(股吧) posts, and East Money's daily popularity rank.

Each source is trimmed to the analysis window. With a TypeSafe key, the social
posts are screened by Jev first (see post_screen). These feeds serve recent items
and are not archived, so a historical run's sentiment inputs are not
point-in-time.

The report is a SentimentReport through structured output where the provider
supports it and free text otherwise, so the band, score and confidence header
reads the same across providers.
"""

from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.post_screen import jev_screen
from tradingagents.agents.schemas import SentimentReport, render_sentiment_report
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.agents.tools import get_news
from tradingagents.dataflows.router import china_auto_enabled
from tradingagents.dataflows.symbols import is_china_a_share
from tradingagents.dataflows.vendors.china.social import fetch_guba_posts, fetch_popularity_rank
from tradingagents.dataflows.vendors.reddit import (
    CRYPTO_SUBREDDITS,
    DEFAULT_SUBREDDITS,
    fetch_reddit_posts,
    subreddits_for,
)
from tradingagents.dataflows.vendors.stocktwits import fetch_stocktwits_messages


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + StockTwits + Reddit data, injects them into the
    prompt as structured blocks, and produces a deterministic sentiment
    report via structured output (with a free-text fallback for providers
    that do not support it).
    """
    structured_llm = bind_structured(llm, SentimentReport, "Sentiment Analyst")

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = get_instrument_context_from_state(state)

        # Pre-fetch all three sources. Each fetcher degrades gracefully and
        # returns a string (no exceptions surface from here), so the LLM
        # always sees something — either real data or a clear placeholder.
        news_block = get_news.func(ticker, start_date, end_date)
        if china_auto_enabled() and is_china_a_share(ticker):
            system_message = _build_a_share_system_message(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                news_block=news_block,
                # Trimmed to the window like the US boards (#1220).
                forum_block=fetch_guba_posts(ticker, start_date=start_date, end_date=end_date),
                rank_block=fetch_popularity_rank(ticker, end_date),
            )
        else:
            # Pass the analysis window so a historical run trims social posts to it
            # instead of leaking today's chatter into a backtest (#1220).
            screen = jev_screen(ticker)
            stocktwits_block = fetch_stocktwits_messages(
                ticker, limit=30, start_date=start_date, end_date=end_date, screen=screen
            )
            subreddits = subreddits_for(ticker)
            reddit_block = fetch_reddit_posts(
                ticker, subreddits, start_date=start_date, end_date=end_date, screen=screen
            )
            system_message = _build_system_message(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                news_block=news_block,
                stocktwits_block=stocktwits_block,
                reddit_block=reddit_block,
                subreddits=subreddits,
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Report what your tools support; another agent decides the trade."
                    # No tool-calling here: the data is pre-fetched into the
                    # prompt, so tool-range wording would only invite a
                    # hallucinated tool call (#1130).
                    " Today's date is {current_date}; treat it as 'now' for all analysis. {instrument_context}"
                    " " + NO_EXTERNAL_TOOLS +
                    "\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # Format the template into a concrete message list so the structured
        # and free-text paths receive the same input. No bind_tools — the
        # data is already in the prompt.
        formatted_messages = prompt.format_messages(messages=state["messages"])

        report_text = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted_messages,
            render_sentiment_report,
            "Sentiment Analyst",
        )

        return {
            "messages": [AIMessage(content=report_text)],
            "sentiment_report": report_text,
        }

    return sentiment_analyst_node


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
    subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS,
) -> str:
    """Assemble the sentiment-analyst system message with structured data blocks."""
    if subreddits == DEFAULT_SUBREDDITS:
        character = "r/wallstreetbets is often contrarian/exuberant; r/stocks more measured; r/investing longer-term"
    elif len(subreddits) > len(CRYPTO_SUBREDDITS):
        character = f"r/{subreddits[0]} leans toward the coin's holders; r/CryptoCurrency and r/CryptoMarkets are broader"
    else:
        character = "r/CryptoCurrency and r/CryptoMarkets are broad crypto communities"
    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, drawing on three complementary data sources that have already been collected for you.

## Data sources (pre-fetched, in this prompt)

### News headlines — Yahoo Finance, past 7 days
Institutional framing. Fact-driven, slower-moving signal.

<start_of_news>
{news_block}
<end_of_news>

### StockTwits messages — retail-trader social platform indexed by cashtag
Fast-moving signal. Each message carries a user-labeled sentiment tag (Bullish / Bearish / no-label) plus the message body.

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### Reddit posts — {", ".join(f"r/{sub}" for sub in subreddits)} (past 7 days)
Community discussion, without vote or comment counts. Subreddit character matters ({character}).

<start_of_reddit>
{reddit_block}
<end_of_reddit>

## How to analyze this data (best practices)

1. **Read the StockTwits Bullish/Bearish ratio as a leading retail-sentiment signal.** A 70/30 bullish/bearish split is moderately bullish; ≥90/10 may indicate over-extension and contrarian risk; 50/50 is uncertainty. Sample size matters — base rates on the actual message count, not percentages alone. A block headed "Screened by Jev" has had off-topic posts removed; its stance count is a classifier's read of every on-topic post fetched, labelled or not, of which the posts listed are a sample. Read it alongside the user tags.

2. **Look for cross-source divergences.** If news framing is bearish but StockTwits is overwhelmingly bullish, that mismatch is itself a signal — it can mean retail is leaning into a thesis the news flow hasn't caught up to (or vice versa, that retail is chasing while institutions are cautious).

3. **Read Reddit posts for substance.** The feed carries no vote or comment counts, so judge a post by its body excerpt, not its title alone, and do not infer engagement.

4. **Distinguish opinion from event.** A news headline ("Nvidia announces $500M Corning deal") is an event; a StockTwits post ("buying NVDA, this is going to moon") is opinion. Both are inputs but should be weighted differently in your conclusions.

5. **Identify recurring narrative themes.** What topic keeps coming up across sources? That's the dominant narrative driving current sentiment.

6. **Be honest about data limits.** If StockTwits returned only a handful of messages, or one or more sources returned an "<unavailable>" placeholder, the sentiment read is less robust — flag this explicitly in the `confidence` field and the narrative. If the sources are silent on a given subreddit, say so.

7. **Identify catalysts and risks** that emerge across sources — news of upcoming earnings, product launches, competitive threats, macro headlines, etc.

8. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call.

{_output_fields()}"""


def _output_fields() -> str:
    return f"""## Output fields

Fill the following fields:

- **overall_band**: Exactly one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. Use Mixed when sources point in clearly different directions; Neutral only when all sources are genuinely silent.
- **overall_score**: A number from 0 (maximally bearish) to 10 (maximally bullish); 5 is neutral. Keep it consistent with overall_band.
- **confidence**: low / medium / high, based on data quality and sample size.
- **narrative**: Full source-by-source breakdown, divergences, dominant narrative themes, catalysts and risks, and a markdown summary table of key sentiment signals (direction, source, supporting evidence).

{get_language_instruction()}"""


def _build_a_share_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    forum_block: str,
    rank_block: str,
) -> str:
    """The sentiment-analyst system message for a China A-share, from Chinese sources."""
    return f"""You are a financial market sentiment analyst covering China A-shares. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, drawing on three complementary data sources that have already been collected for you.

## Data sources (pre-fetched, in this prompt)

### News and exchange announcements — East Money news and CNINFO announcements
Institutional framing and the company's own disclosures. Announcements (results, risk warnings, regulatory letters, contracts, shareholding changes) are facts; news articles are framing.

<start_of_news>
{news_block}
<end_of_news>

### East Money stock forum (股吧) — the largest Chinese retail board for individual stocks
Fast-moving retail opinion. Each post carries its publish time, view count and reply count, but no sentiment label: read the tone of each title yourself.

<start_of_forum>
{forum_block}
<end_of_forum>

### East Money popularity rank (人气榜) — daily rank among all A-shares by retail attention
1 is the most-watched stock that day. A sharp climb means retail attention is arriving; a fall means it is leaving.

<start_of_rank>
{rank_block}
<end_of_rank>

## How to analyze this data (best practices)

1. **Classify the forum titles.** Count roughly how many read bullish (抄底, 起飞, 涨停, 加仓), bearish (割肉, 跌停, 退市, 跑路, 套牢) or neutral, and report the split with the sample size. Weight posts with many views and replies above posts nobody read.

2. **Read attention and tone together.** A rising popularity rank with euphoric titles is crowding and contrarian risk; a rising rank with angry or fearful titles is capitulation or a bad-news shock. Rank alone says attention, not direction.

3. **Look for divergence between disclosures and chatter.** If announcements are negative (risk warning, regulatory letter, loss warning) while the forum is euphoric, or the reverse, that gap is itself a signal.

4. **Weigh risk-warning context.** For an ST or *ST stock, forum talk about 摘帽 (removal of the warning) or 退市 (delisting) dominates sentiment; say which way it leans and whether any announcement supports it.

5. **Identify recurring narrative themes** (a contract, a concept such as AI or compute, a restructuring, a regulator) and whether the forum is trading the theme or the fundamentals.

6. **Be honest about data limits.** Forum titles are short and noisy, and the forum serves only recent posts; if a source returned an "<unavailable>" placeholder or few posts, flag it in the `confidence` field and the narrative.

7. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call.

{_output_fields()}"""
