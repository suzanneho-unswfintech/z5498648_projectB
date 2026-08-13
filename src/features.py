"""
Station 2 - Feature Engineering & Text Assembly (descriptive parts)
FINS3645 Project B - reused verbatim from Project A (Stations 1-2).

This is the validated Part A feature layer, copied unchanged. The methodology
and function interfaces are identical to Part A; only this header note differs.

NOTE: this file computes descriptive/exploratory features only. Sentiment
SCORING (VADER or otherwise) belongs in src/sentiment.py (Station 3), not here
- the descriptive/scoring split is unchanged from Part A. Counting sentiment-
bearing VOCABULARY (how many words in a headline appear in a lexicon) is a
word count, not a sentiment score, so that stays here.
"""
from __future__ import annotations

import pandas as pd

# Comprehensive general-English stopword list (~170 words), used as the
# DEFAULT for top_terms() so casual function words (the, of, in, will,
# after, its...) don't dominate a word-frequency exhibit. Callers can still
# pass additional domain-specific stopwords (e.g. "says", "reuters") which
# get UNIONED with this list, not used instead of it - see top_terms().
ENGLISH_STOPWORDS = frozenset({
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","aren't","as","at","be","because","been","before","being","below",
    "between","both","but","by","can","can't","cannot","could","couldn't",
    "did","didn't","do","does","doesn't","doing","don","don't","down","during",
    "each","few","for","from","further","had","hadn't","has","hasn't","have",
    "haven't","having","he","he'd","he'll","he's","her","here","here's","hers",
    "herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've",
    "if","in","into","is","isn't","it","it's","its","itself","just","let's",
    "me","more","most","mustn't","my","myself","no","nor","not","of","off",
    "on","once","only","or","other","ought","our","ours","ourselves","out",
    "over","own","s","same","shan't","she","she'd","she'll","she's","should",
    "shouldn't","so","some","such","t","than","that","that's","the","their",
    "theirs","them","themselves","then","there","there's","these","they",
    "they'd","they'll","they're","they've","this","those","through","to",
    "too","under","until","up","very","was","wasn't","we","we'd","we'll",
    "we're","we've","were","weren't","what","what's","when","when's","where",
    "where's","which","while","who","who's","whom","why","why's","will",
    "with","won't","would","wouldn't","you","you'd","you'll","you're","you've",
    "your","yours","yourself","yourselves","also","after","amid","said","says",
    "new","one","two","first","its","it's",
})


# ---------------------------------------------------------------------------
# Dataset inventory (required exact filename: dataset_inventory.csv)
# ---------------------------------------------------------------------------

def dataset_inventory(equity: pd.DataFrame, crypto: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    def _row(name, df, date_col="date"):
        return {
            "dataset": name,
            "n_rows": len(df),
            "n_tickers": df["ticker"].nunique() if "ticker" in df.columns else None,
            "start_date": df[date_col].min(),
            "end_date": df[date_col].max(),
            "frequency": None,  # filled in below per dataset
            "coverage": None,   # filled in below per dataset
        }

    n_eq_days = equity["date"].nunique()
    n_eq_tickers = equity["ticker"].nunique()
    n_eq_sectors = equity["sector"].nunique() if "sector" in equity.columns else None
    n_cr_days = crypto["date"].nunique()
    n_cr_tickers = crypto["ticker"].nunique()
    n_news_tickers = news["ticker"].nunique() if "ticker" in news.columns else None
    n_news_sectors = news["sector"].nunique() if "sector" in news.columns else None
    n_news_days = news["date"].nunique()

    rows = [_row("equity_prices", equity), _row("crypto_prices", crypto), _row("news_headlines", news)]
    rows[0]["frequency"] = "daily, business days (~252/yr)"
    rows[0]["coverage"] = f"{n_eq_days:,} trading days x {n_eq_tickers} tickers, {n_eq_sectors} sectors"
    rows[1]["frequency"] = "daily, every day (~365/yr)"
    rows[1]["coverage"] = f"{n_cr_days:,} calendar days x {n_cr_tickers} coins (price-only, no sector)"
    rows[2]["frequency"] = "irregular, multiple headlines per ticker-day possible"
    rows[2]["coverage"] = f"{n_news_tickers} tickers, {n_news_sectors} sectors, {n_news_days:,} calendar days span"
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Descriptive stats of returns by asset class (required exact filename:
# descriptive_stats_returns.csv)
# ---------------------------------------------------------------------------

def descriptive_stats_returns(equity_ret: pd.DataFrame, crypto_ret: pd.DataFrame) -> pd.DataFrame:
    def _stats(df, label):
        r = df["ret"].dropna()
        return {
            "asset_class": label,
            "n_obs": len(r),
            "mean_daily": r.mean(),
            "vol_daily": r.std(),
            "min": r.min(),
            "max": r.max(),
            "skew": r.skew(),
            "kurtosis": r.kurt(),
        }
    return pd.DataFrame([_stats(equity_ret, "equity"), _stats(crypto_ret, "crypto")])


# ---------------------------------------------------------------------------
# Text descriptive exploration (article volume + vocabulary counts only)
# ---------------------------------------------------------------------------

def article_counts_over_time(news: pd.DataFrame, freq: str = "ME") -> pd.Series:
    """Number of headlines per calendar period (default monthly)."""
    return news.set_index("date")["title"].resample(freq).count()


def load_vader_lexicon() -> set[str]:
    """Load the VADER lexicon's vocabulary (word list only, no polarity
    weights used) for word-COUNTING purposes. Requires `vaderSentiment`
    to be installed (`pip install vaderSentiment`) - it's the standard
    lexicon this course expects for Part B, so installing it now is not
    scope creep. Falls back to a small hand-built list with a loud
    warning if the package isn't available, so the pipeline doesn't
    silently produce near-zero counts.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return set(SentimentIntensityAnalyzer().lexicon.keys())
    except ImportError:
        import warnings
        warnings.warn(
            "vaderSentiment not installed - falling back to a tiny placeholder "
            "lexicon. Run `pip install vaderSentiment` for a real word count "
            "(it's needed for Part B anyway)."
        )
        return {
            "good", "great", "strong", "up", "gain", "gains", "growth", "profit",
            "bad", "weak", "down", "loss", "losses", "decline", "cut", "crisis",
            "beat", "miss", "surge", "plunge", "rally", "slump",
        }


def sentiment_word_count_summary(news: pd.DataFrame, lexicon: set[str]) -> pd.DataFrame:
    """Aggregate, per sector, the average number of lexicon-matched words
    per headline and the average headline length - a DESCRIPTIVE summary
    of how much sentiment-bearing vocabulary appears in the news flow.
    Not a polarity score (that's Part B)."""
    scored = sentiment_word_counts(news, lexicon)
    scored = scored.merge(news[["ticker", "date", "sector"]].drop_duplicates(["ticker", "date"]),
                           on=["ticker", "date"], how="left")
    summary = (scored.groupby("sector")
               .agg(n_headlines=("title", "count"),
                    avg_words_per_headline=("n_words", "mean"),
                    avg_lexicon_words_per_headline=("n_lexicon_words", "mean"))
               .reset_index()
               .sort_values("avg_lexicon_words_per_headline", ascending=False))
    return summary


def rolling_volatility(ret_df: pd.DataFrame, window: int = 20, annualise_factor: int = 252) -> pd.DataFrame:
    """Rolling annualised volatility per ticker.

    Equation: vol_t = std(ret[t-window+1 : t]) * sqrt(annualise_factor)

    Why useful: daily return std alone is noisy and not comparable across
    assets with different trading calendars; annualising puts equity and
    crypto on a comparable scale, and the rolling window shows how risk
    changes over time rather than one static number for the whole sample.
    Not an input to the optimiser (brief) - purely descriptive here.

    annualise_factor: use 252 for equity (trading days/yr), 365 for crypto
    (calendar days/yr) - pass the value matching the panel you call this on.
    """
    df = ret_df.sort_values(["ticker", "date"]).copy()
    df["rolling_vol"] = (
        df.groupby("ticker")["ret"]
        .transform(lambda s: s.rolling(window, min_periods=window).std() * (annualise_factor ** 0.5))
    )
    return df


def rolling_sharpe(ret_df: pd.DataFrame, window: int = 20, annualise_factor: int = 252,
                    risk_free_daily: float = 0.0) -> pd.DataFrame:
    """Rolling annualised Sharpe ratio per ticker.

    Equation: Sharpe_t = (mean(ret[t-window+1:t]) - rf) / std(ret[t-window+1:t])
              * sqrt(annualise_factor)

    Why useful: raw return alone doesn't say whether the return was earned
    with a lot of risk or a little - Sharpe normalises return per unit of
    volatility, which is the natural way to compare individual assets
    before any portfolio construction (Part B). risk_free_daily defaults
    to 0 since no risk-free series is provided in the data bundle; noted
    as a simplifying assumption, not an oversight.
    """
    df = ret_df.sort_values(["ticker", "date"]).copy()

    def _sharpe(s):
        mu = s.rolling(window, min_periods=window).mean() - risk_free_daily
        sigma = s.rolling(window, min_periods=window).std()
        return (mu / sigma) * (annualise_factor ** 0.5)

    df["rolling_sharpe"] = df.groupby("ticker")["ret"].transform(_sharpe)
    return df


def drawdown_series(ret_df: pd.DataFrame) -> pd.DataFrame:
    """Drawdown series per ticker, from daily returns.

    Equations:
        Wealth_t   = cumprod(1 + ret_t), starting Wealth_0 = 1
        Peak_t     = running max of Wealth up to t
        Drawdown_t = Wealth_t / Peak_t - 1   (<= 0 always; 0 = at a new high)

    Why useful: a single "max drawdown" number tells you the worst peak-
    to-trough loss, but the drawdown SERIES shows how long the asset
    spent underwater and how many separate drawdown episodes it had -
    two assets with the same max drawdown can have very different
    "time spent recovering" profiles, which matters more to a long-term
    holder than the single worst number alone.
    """
    df = ret_df.sort_values(["ticker", "date"]).copy()

    def _dd(g):
        wealth = (1 + g["ret"].fillna(0)).cumprod()
        peak = wealth.cummax()
        return wealth / peak - 1

    df["drawdown"] = df.groupby("ticker", group_keys=False).apply(_dd)
    return df


def max_drawdown_table(ret_df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker summary: max drawdown, the date it happened, and the
    prior peak date it fell from. Equation for max drawdown itself:
    MaxDrawdown = min(Drawdown_t) over the whole sample (see
    drawdown_series() for how Drawdown_t is built).
    """
    dd = drawdown_series(ret_df)
    rows = []
    for ticker, g in dd.groupby("ticker"):
        trough_idx = g["drawdown"].idxmin()
        trough_row = g.loc[trough_idx]
        # peak date = the last date before the trough where drawdown was 0
        pre_trough = g.loc[:trough_idx]
        peak_date = pre_trough.loc[pre_trough["drawdown"] == 0, "date"].max()
        rows.append({
            "ticker": ticker,
            "max_drawdown": trough_row["drawdown"],
            "trough_date": trough_row["date"],
            "peak_date_before_trough": peak_date,
        })
    return pd.DataFrame(rows).sort_values("max_drawdown")  # most severe first (most negative)


def news_coverage_matrix(news: pd.DataFrame) -> pd.DataFrame:
    """Ticker x month headline-count matrix, for a coverage heatmap.

    Why useful: the dataset-inventory table only shows AGGREGATE headline
    counts; it doesn't show whether coverage is even across tickers and
    time. A ticker with very sparse coverage in some months is a real
    limitation for any sentiment-based fund built on it in Part B - this
    exhibit surfaces that before it becomes a hidden problem later.
    """
    df = news.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    matrix = df.groupby(["ticker", "month"]).size().unstack(fill_value=0)
    return matrix.sort_index()


def load_vader_pos_neg_lexicons() -> tuple[set[str], set[str]]:
    """Split the VADER lexicon into (positive_words, negative_words) sets
    by the sign of each word's valence score. Used ONLY to count how many
    positive-vocabulary and negative-vocabulary words appear in a headline
    - two separate counts, never combined into a net/compound score. A
    net score would functionally BE a sentiment index, which the brief
    reserves for Part B; two raw vocabulary counts stay descriptive.
    Falls back to a small hand-built pos/neg split if vaderSentiment isn't
    installed (same fallback policy as load_vader_lexicon).
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        lexicon = SentimentIntensityAnalyzer().lexicon
        positive = {w for w, v in lexicon.items() if v > 0}
        negative = {w for w, v in lexicon.items() if v < 0}
        return positive, negative
    except ImportError:
        import warnings
        warnings.warn(
            "vaderSentiment not installed - falling back to a tiny placeholder "
            "pos/neg lexicon. Run `pip install vaderSentiment` for real counts."
        )
        positive = {"good", "great", "strong", "up", "gain", "gains", "growth",
                    "profit", "beat", "surge", "rally"}
        negative = {"bad", "weak", "down", "loss", "losses", "decline", "cut",
                    "crisis", "miss", "plunge", "slump"}
        return positive, negative


def positive_negative_word_counts(news: pd.DataFrame) -> pd.DataFrame:
    """Per-headline count of positive-vocabulary and negative-vocabulary
    words (VADER lexicon, split by valence sign). Descriptive counting
    only - see load_vader_pos_neg_lexicons() docstring for why this stops
    short of a net/compound sentiment score."""
    positive, negative = load_vader_pos_neg_lexicons()
    out = news.copy()
    tokens = out["title"].str.lower().str.findall(r"[a-z']+")
    out["n_positive_words"] = tokens.apply(lambda toks: sum(1 for t in toks if t in positive))
    out["n_negative_words"] = tokens.apply(lambda toks: sum(1 for t in toks if t in negative))
    return out[["ticker", "date", "sector", "title", "n_positive_words", "n_negative_words"]]


def positive_negative_summary_by_sector(news: pd.DataFrame) -> pd.DataFrame:
    """Aggregate positive/negative word counts by sector - two columns,
    side by side, so the report can show which sectors' headlines lean on
    more positive vs negative vocabulary WITHOUT computing a combined
    score (that combination step is Part B's job)."""
    scored = positive_negative_word_counts(news)
    return (scored.groupby("sector")
            .agg(n_headlines=("title", "count"),
                 avg_positive_words=("n_positive_words", "mean"),
                 avg_negative_words=("n_negative_words", "mean"))
            .reset_index()
            .sort_values("avg_positive_words", ascending=False))


def sentiment_word_counts(news: pd.DataFrame, lexicon: set[str]) -> pd.DataFrame:
    """Count, per headline, how many words appear in a given lexicon.

    This is a DESCRIPTIVE vocabulary count, not a sentiment score - it does
    not sum polarity weights or classify the headline as positive/negative.
    That step is Part B. `lexicon` should be a plain set of lowercase words
    (e.g. the VADER lexicon keys) supplied by the caller.
    """
    out = news.copy()
    out["n_words"] = out["title"].str.split().str.len()
    out["n_lexicon_words"] = out["title"].str.lower().str.findall(r"[a-z']+").apply(
        lambda toks: sum(1 for t in toks if t in lexicon)
    )
    return out[["ticker", "date", "title", "n_words", "n_lexicon_words"]]


def top_terms(news: pd.DataFrame, n: int = 30, stopwords: set[str] | None = None,
              min_word_length: int = 3) -> pd.Series:
    """Simple word-frequency count across all headlines, for a word-cloud
    or bar-chart exhibit. Always excludes ENGLISH_STOPWORDS (a general
    stopword list) - `stopwords` is for ADDITIONAL domain-specific words
    to exclude on top of that, not a replacement for it. Also drops
    fragments shorter than min_word_length (default 3) - these are
    almost always apostrophe artefacts (e.g. "s" from "company's") or
    single letters, not real content words. Also excludes the tickers
    themselves (lowercased) since a word cloud dominated by "aapl" or
    "nvda" repeating the row's own ticker isn't informative.
    """
    extra_stopwords = stopwords or set()
    tickers_lower = set(news["ticker"].str.lower().unique()) if "ticker" in news.columns else set()
    all_stopwords = ENGLISH_STOPWORDS | extra_stopwords | tickers_lower

    all_words = news["title"].str.lower().str.findall(r"[a-z']+").explode()
    all_words = all_words[all_words.str.len() >= min_word_length]
    all_words = all_words[~all_words.isin(all_stopwords)]
    return all_words.value_counts().head(n)
