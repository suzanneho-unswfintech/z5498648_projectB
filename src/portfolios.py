"""Station 3 - your funds: optimal portfolios + out-of-sample backtest.

Build at least a combined equity-plus-crypto fund with two optimisation methods.
Backtest rules: walk-forward, no look-ahead, weights from past data only, annualise
with 252 (equity) or 365 (crypto). See the brief, Part B.

Assumptions made here (record these in the report, they are not derivable from
the data and change the numbers if changed):
- Long-only: every weight function constrains weights to sum to 1 and be >= 0.
  No short-selling in any of the four methods.
- Risk-free rate = 0 in both max_sharpe's Sharpe objective and
  performance_metrics' Sharpe ratio. No risk-free series is provided in the
  data bundle; 0 is a simplifying assumption, not a gap - same choice already
  made in Part A's rolling_sharpe() for consistency across both parts.
- Initial training window = calendar-based, 2020-01-01 through 2022-12-31
  (NOT a fixed day-count window). OOS period = everything after that, which
  given the data's 2023-12-31 cap means OOS = 2023 only, about 12 monthly
  rebalances. This trades away OOS sample size for a longer, calendar-clean
  initial estimation period - a deliberate choice, note it in the report.
- Retrain frequency = monthly (first trading day of each calendar month in
  the OOS period), using an EXPANDING window each time: all data from
  2020-01-01 up to (never including) the rebalance date. Later rebalances
  are therefore trained on strictly more data than earlier ones - the
  strategies are not on a like-for-like footing across the OOS period. This
  is an explicit choice (vs. a fixed rolling window), not an oversight.
- Weights are held FLAT between rebalances (no drift adjustment for realised
  returns moving the portfolio away from target weights intra-month). This is
  the standard simplifying convention for a teaching-scale backtest; note it
  in the report as a limitation, not hidden.
- An asset is excluded from a given rebalance's window only if it has a
  GAP after its own first observed return (or hasn't started trading in the
  window at all) - not merely because its first-ever return is NaN (that is
  structural, from pct_change, and would otherwise wrongly and permanently
  exclude assets from an expanding window - see _clean_window()'s docstring).
- Transaction costs (extension, see apply_transaction_costs): 5 basis points
  (0.05%) per rebalance, charged on ONE-WAY portfolio turnover, cost paid at
  the start of the rebalance day before that day's returns accrue. The
  zero-transaction-cost backtest above is unchanged and remains the baseline;
  the cost model is applied ON TOP of it as a reusable wrapper. The initial
  cash-to-weights purchase on the first OOS date is excluded from turnover
  (include_initial=False) so every fund is charged for the same thing -
  rebalancing, not the one-off initial allocation.
- CVaR method (extension, see cvar_min): minimises the empirical CVaR of daily
  portfolio LOSSES at confidence alpha=0.95 (expected loss in the worst 5% of
  training-window days) via the Rockafellar-Uryasev LP, long-only and fully
  invested, on the same expanding strictly-past windows as the other methods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from src import etl


def build_universe_returns(universe: str) -> pd.DataFrame:
    """Return a wide daily-returns panel (date index, one column per asset)
    for the requested universe, built from the raw panels via src.etl.

    universe: "equity" (bare ticker columns), "crypto" (bare ticker columns),
    or "combined" (EQ_*/CR_* prefixed, on the equity trading calendar).

    Returns are always computed WITHIN each panel before any merge - price
    levels are never merged first, so no spurious seam returns are created.
    """
    if universe not in ("equity", "crypto", "combined"):
        raise ValueError(f"universe must be 'equity', 'crypto' or 'combined', got {universe!r}")

    raw = etl.load_raw()

    if universe in ("equity", "combined"):
        equity_clean, _ = etl.clean_equity(raw["equity"])
        equity_ret = etl.compute_returns(equity_clean)

    if universe in ("crypto", "combined"):
        crypto_clean, _ = etl.clean_crypto(raw["crypto"])
        crypto_ret = etl.compute_returns(crypto_clean)

    if universe == "equity":
        return equity_ret.pivot(index="date", columns="ticker", values="ret").sort_index()

    if universe == "crypto":
        return crypto_ret.pivot(index="date", columns="ticker", values="ret").sort_index()

    return etl.merge_calendars(equity_ret, crypto_ret)


# ---------------------------------------------------------------------------
# Rebalance schedule
# ---------------------------------------------------------------------------

def get_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First trading day of each calendar month present in `index`.

    Called once per backtest and shared across all methods/universes, so the
    rebalance schedule can't silently drift between funds being compared.
    """
    index = pd.DatetimeIndex(index)
    s = pd.Series(index=index, data=index)
    first_per_month = s.groupby(index.to_period("M")).min()
    return pd.DatetimeIndex(sorted(first_per_month.values))


# ---------------------------------------------------------------------------
# Weight functions - each takes a trailing window (assets as columns, no
# NaN - oos_backtest guarantees this) and returns weights summing to 1.
# ---------------------------------------------------------------------------

def equal_weight(window: pd.DataFrame) -> pd.Series:
    """1/N across every asset present in the window."""
    n = window.shape[1]
    return pd.Series(1.0 / n, index=window.columns)


def min_variance(window: pd.DataFrame) -> pd.Series:
    """Long-only global minimum-variance portfolio: minimise w'Sigma w
    subject to sum(w) = 1, w >= 0.

    The objective is scaled up (SCALE) and ftol tightened below SLSQP's
    default: daily-return variances are ~1e-4 in magnitude, and SLSQP's
    default convergence tolerance (1e-6) is coarse relative to that, so the
    unscaled optimizer reports success after zero real movement from the
    initial equal-weight guess. Verified against synthetic data: without
    this fix, min_variance silently returned equal weight regardless of
    input (confirmed by result.success=True at x=w0 exactly).
    """
    cov = window.cov().values
    n = cov.shape[0]
    w0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    SCALE = 1e4
    result = minimize(lambda w: (w @ cov @ w) * SCALE, w0, bounds=bounds, constraints=constraints,
                       method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
    w = result.x if result.success else w0
    w = np.clip(w, 0.0, None)
    w = w / w.sum()
    return pd.Series(w, index=window.columns)


def max_sharpe(window: pd.DataFrame, rf: float = 0.0) -> pd.Series:
    """Long-only tangency portfolio: maximise (w'mu - rf) / sqrt(w'Sigma w),
    subject to sum(w) = 1, w >= 0. rf defaults to 0 (see module docstring)."""
    mu = window.mean().values
    cov = window.cov().values
    n = len(mu)
    w0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def neg_sharpe(w):
        port_ret = w @ mu - rf
        port_vol = np.sqrt(w @ cov @ w)
        return -port_ret / port_vol if port_vol > 1e-10 else 0.0

    result = minimize(neg_sharpe, w0, bounds=bounds, constraints=constraints, method="SLSQP")
    w = result.x if result.success else w0
    w = np.clip(w, 0.0, None)
    w = w / w.sum() if w.sum() > 0 else w0
    return pd.Series(w, index=window.columns)


def risk_parity(window: pd.DataFrame) -> pd.Series:
    """Equal risk contribution portfolio: each asset contributes 1/N of
    total portfolio variance.

    Uses the convex reformulation (Spinu, 2013): minimise
    0.5 * w'Sigma w - (1/n) * sum(log w), w > 0, then normalise w to sum
    to 1 (valid because the unconstrained-sum solution is scale-consistent -
    normalising preserves the equal risk-contribution PERCENTAGES).

    This replaces an earlier direct formulation (minimise the sum of squared
    deviations of each asset's risk-contribution share from 1/N) that looked
    reasonable but is non-convex and got stuck in a bad local minimum on
    synthetic test data - one asset pinned near its lower bound with the
    other three unequally sharing risk (0%, 33%, 33%, 33% instead of the
    correct 25% each). The log-barrier form here is provably convex, so any
    local solver finds the true global optimum - verified on the same
    synthetic data (all four risk-contribution shares came out at exactly
    0.25).
    """
    cov = window.cov().values
    n = cov.shape[0]
    w0 = np.repeat(1.0 / n, n)

    def objective(w):
        return 0.5 * w @ cov @ w - (1.0 / n) * np.sum(np.log(w))

    bounds = [(1e-8, None)] * n
    result = minimize(objective, w0, bounds=bounds, method="SLSQP",
                      options={"ftol": 1e-14, "maxiter": 1000})
    w = result.x if result.success else w0
    w = np.clip(w, 1e-12, None)
    w = w / w.sum()
    return pd.Series(w, index=window.columns)


def cvar_min(window: pd.DataFrame, alpha: float = 0.95, max_weight: float = 1.0) -> pd.Series:
    """Long-only minimum-CVaR portfolio: minimise the empirical CVaR of daily
    portfolio LOSSES at confidence alpha, subject to sum(w) = 1 and 0 <= w
    <= max_weight.

    CVaR_alpha (expected shortfall) is the expected loss conditional on the
    loss being in the worst (1 - alpha) tail of the training-window return
    distribution. Minimising it sizes the fund against its worst 5% of
    historical days (alpha=0.95) instead of its variance.

    Uses the exact Rockafellar-Uryasev (2000) linear reformulation:

        min_{w, beta, z} beta + 1/((1-alpha) * T) * sum_t z_t
        s.t.  z_t >= -(w'R_t) - beta,  z_t >= 0,  beta free,
              1'w = 1,  0 <= w <= max_weight

    over the T historical return scenarios R_t. At the optimum beta is the
    VaR of the loss distribution and the objective equals CVaR. Solved as an
    LP with scipy's highs method: an exact convex solver, unlike the SLSQP
    optimisers above it, so it cannot silently stall on a flat objective.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}")
    if not 0.0 < max_weight <= 1.0:
        raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")

    R = window.dropna(axis=0).values
    n = window.shape[1]
    T = R.shape[0]
    w0 = np.repeat(1.0 / n, n)

    if T == 0 or n == 0:
        return pd.Series(w0, index=window.columns)

    # Variables: x = [w (n), beta (1), z (T)].
    n_z = T
    z_col = n + 1

    # Scenario constraints: -(w'R_t) - beta - z_t <= 0.
    A_ub = np.zeros((T, n + 1 + n_z))
    A_ub[:, :n] = -R
    A_ub[:, n] = -1.0
    for t in range(T):
        A_ub[t, z_col + t] = -1.0
    b_ub = np.zeros(T)

    # Fully invested: sum(w) = 1.
    A_eq = np.zeros((1, n + 1 + n_z))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    # Objective: beta + 1/((1-alpha) * T) * sum(z_t).
    c = np.zeros(n + 1 + n_z)
    c[n] = 1.0
    c[z_col:] = 1.0 / ((1.0 - alpha) * T)

    bounds = [(0.0, max_weight)] * n + [(None, None)] + [(0.0, None)] * n_z

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                     method="highs")
    w = result.x[:n] if result.success else w0
    w = np.clip(w, 0.0, None)
    w = w / w.sum() if w.sum() > 0 else w0
    return pd.Series(w, index=window.columns)


METHODS = {
    "equal_weight": equal_weight,
    "min_variance": min_variance,
    "max_sharpe": max_sharpe,
    "risk_parity": risk_parity,
    "cvar_min": cvar_min,
}


# ---------------------------------------------------------------------------
# Walk-forward out-of-sample backtest
# ---------------------------------------------------------------------------

TRAIN_END_DEFAULT = pd.Timestamp("2022-12-31")  # initial window: 2020-01-01 through this date


def _clean_window(hist: pd.DataFrame) -> pd.DataFrame:
    """Keep an asset in the window only if, from its OWN first observed
    return onward, it has no gaps through the end of the window.

    This is deliberately NOT `hist.dropna(axis=1, how="any")`. That naive
    check would permanently exclude almost every asset from an EXPANDING
    window: compute_returns()'s pct_change() makes an asset's very first
    return NaN by construction (no prior price to diff against), and once
    the window starts from the beginning of the data, that one structural
    NaN never ages out - it sits in every window forever. For the combined
    equity+crypto universe specifically, equity's first-day NaN and crypto's
    first-day NaN don't fall on the same calendar date (crypto trades every
    day, equity only business days), so a naive dropna would see equity's
    lone leading NaN sitting in every window and silently exclude all EQ_*
    columns from every rebalance, forever - turning a "combined" backtest
    into a crypto-only one with no error or warning. Confirmed via a
    targeted synthetic test before this fix (see tests/test_portfolios.py).

    Leading NaN before an asset's own first trade is expected and NOT a
    reason to exclude it. A genuine gap AFTER it starts (or it never
    appearing in the window at all) is a reason to exclude it.
    """
    keep_cols = []
    for col in hist.columns:
        s = hist[col].dropna()
        if len(s) == 0:
            continue  # asset has no data at all in this window yet
        first_valid = s.index.min()
        segment = hist.loc[first_valid:, col]
        if segment.isna().any():
            continue  # a real gap after the asset started - exclude this window
        keep_cols.append(col)
    return hist[keep_cols]


def oos_backtest(returns: pd.DataFrame, method: str = "min_variance",
                  train_end: pd.Timestamp = TRAIN_END_DEFAULT) -> dict:
    """Walk-forward OOS backtest for one universe x one method.

    Initial training window: all data from the start of `returns` through
    `train_end` (calendar-based, default 2020-01-01 to 2022-12-31 - NOT a
    fixed day-count window). OOS period: everything after `train_end`.

    Retrain frequency: monthly (first trading day of each month in the OOS
    period). Each retrain uses an EXPANDING window - all data from the very
    start of `returns` up to (never including) the rebalance date - so later
    rebalances are trained on strictly more data than earlier ones. See the
    module docstring for why this is a deliberate choice.

    Look-ahead safety: at rebalance date `t`, the window is
    returns.loc[returns.index < t] - strictly before `t`, never including
    it - so this is safe by construction, not just by convention. Weights
    are held flat (unchanged) between rebalances.

    Returns a dict: daily_returns (pd.Series), weights (pd.DataFrame, one
    row per OOS date), growth_of_1 (pd.Series).
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, must be one of {list(METHODS)}")
    weight_fn = METHODS[method]

    dates = returns.index
    if not (dates <= train_end).any():
        raise ValueError(f"no data on or before train_end={train_end}; "
                         f"can't build the initial window")

    oos_dates = dates[dates > train_end]
    if len(oos_dates) == 0:
        raise ValueError(f"no data after train_end={train_end}; nothing to backtest out-of-sample")

    rebalance_dates = get_rebalance_dates(oos_dates)
    # always rebalance on the first OOS date too, even if it isn't the 1st
    # of a month, so weights are defined from day one of the OOS period
    if len(rebalance_dates) == 0 or rebalance_dates[0] != oos_dates[0]:
        rebalance_dates = pd.DatetimeIndex([oos_dates[0]]).append(rebalance_dates)

    assets = returns.columns
    weights_over_time = pd.DataFrame(0.0, index=oos_dates, columns=assets)
    current_weights = pd.Series(0.0, index=assets)

    for date in oos_dates:
        if date in rebalance_dates:
            hist = returns.loc[returns.index < date]  # expanding: everything strictly before `date`
            hist_clean = _clean_window(hist)
            if hist_clean.shape[1] == 0:
                raise ValueError(f"no assets with usable history at {date}")
            w = weight_fn(hist_clean)
            current_weights = pd.Series(0.0, index=assets)
            current_weights.loc[w.index] = w.values
        weights_over_time.loc[date] = current_weights

    # fillna(0) on returns here is safe, not silent data loss: it only ever
    # multiplies against a weight of exactly 0 (an asset excluded from that
    # window), so it can't hide a real return being lost.
    daily_returns = (weights_over_time * returns.loc[oos_dates].fillna(0.0)).sum(axis=1)
    growth_of_1 = (1.0 + daily_returns).cumprod()

    return {
        "daily_returns": daily_returns,
        "weights": weights_over_time,
        "growth_of_1": growth_of_1,
    }


# ---------------------------------------------------------------------------
# Fact-sheet metrics
# ---------------------------------------------------------------------------

def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Annualised return, annualised volatility, Sharpe (rf=0, see module
    docstring), and max drawdown from a daily portfolio return series."""
    r = daily_returns.dropna()
    n = len(r)
    if n == 0:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}

    ann_return = (1.0 + r).prod() ** (periods_per_year / n) - 1.0
    ann_vol = r.std() * np.sqrt(periods_per_year)
    # epsilon, not > 0: a near-constant return series has floating-point std()
    # noise (~1e-18) that is technically nonzero, which would otherwise blow
    # Sharpe up to ~1e16 instead of correctly reporting it as undefined.
    sharpe = ann_return / ann_vol if ann_vol > 1e-10 else np.nan

    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    max_dd = drawdown.min()

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


# ---------------------------------------------------------------------------
# Transaction-cost model (extension - applied on top of the zero-cost baseline)
# ---------------------------------------------------------------------------

def portfolio_turnover(weights: pd.DataFrame, include_initial: bool = False) -> pd.Series:
    """One-way turnover per date: turnover(t) = 0.5 * sum_i |w_i(t) - w_i(t-1)|.

    Why one-way (the 0.5): a long-only, fully invested fund that sells X of
    asset A to buy X of asset B trades X, not 2X - what is sold funds what is
    bought, so half the gross absolute weight change equals the amount traded.

    `weights` is the wide weight frame from oos_backtest() (one row per OOS
    date, held flat between rebalances), so turnover is nonzero only on the
    dates where the weights actually change (rebalance dates).

    include_initial: whether the first date's move from cash (0 weights) to
    the initial target weights counts as a trade. Default False - excluding
    it keeps every fund charged for the same thing (rebalancing, not the
    one-off initial allocation), so cross-fund comparison is like-for-like.
    """
    if include_initial:
        prev = weights.shift(1).fillna(0.0)
        turnover = weights.sub(prev).abs().sum(axis=1) / 2.0
    else:
        # diff() first row is NaN -> treated as 0 turnover (no prior weights)
        turnover = weights.diff().abs().sum(axis=1) / 2.0
    return turnover.fillna(0.0)


def transaction_cost(turnover: pd.Series, cost_rate: float = 0.0005) -> pd.Series:
    """Dollar-cost of trading as a fraction of NAV: cost(t) = turnover(t) * rate.

    cost_rate: 0.0005 = 5 basis points per unit of one-way turnover. Charged
    only on dates with nonzero turnover (rebalance dates).
    """
    return turnover * cost_rate


def apply_transaction_costs(gross_returns: pd.Series, weights: pd.DataFrame,
                            cost_rate: float = 0.0005,
                            include_initial: bool = False) -> pd.DataFrame:
    """Layer the transaction-cost model onto an existing gross return series.

    gross_returns and weights must be aligned (same index - exactly what
    oos_backtest() returns for one fund: daily_returns and weights both on
    the OOS dates). The zero-cost baseline is not recomputed here; this only
    subtracts the cost from it.

    Net return on a rebalance day t:
        net(t) = (1 + gross(t)) * (1 - cost(t)) - 1
    The cost is paid at the START of the rebalance day, before that day's
    returns accrue, so the day's return is applied to the post-cost value.
    On non-rebalance days cost = 0 and net = gross.

    Returns a DataFrame [gross_return, turnover, cost, net_return] indexed
    like gross_returns - the turnover/cost columns are the exhibit inputs
    for a turnover-over-time or cost comparison table.
    """
    missing = set(weights.index).difference(gross_returns.index)
    if missing:
        raise ValueError(
            f"weights contains {len(missing)} date(s) not in gross_returns "
            f"(e.g. {sorted(missing)[:3]}); align the two series first"
        )
    if len(weights) != len(gross_returns):
        raise ValueError(
            f"weights and gross_returns must have the same length, got "
            f"{len(weights)} vs {len(gross_returns)}"
        )

    turnover = portfolio_turnover(weights, include_initial=include_initial)
    cost = transaction_cost(turnover, cost_rate).reindex(gross_returns.index, fill_value=0.0)
    net = (1.0 + gross_returns) * (1.0 - cost) - 1.0

    return pd.DataFrame({
        "gross_return": gross_returns,
        "turnover": turnover.reindex(gross_returns.index, fill_value=0.0),
        "cost": cost,
        "net_return": net,
    })
