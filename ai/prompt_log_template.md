# Prompt log - Project B


## Entry 1 - Reuse Part A etl.py and features.py in Project B
### What I wanted
Reuse my validated Part A data foundation in Project B instead of rewriting it:
copy `src/etl.py` and `src/features.py` from Project A with the cleaning
methodology, feature engineering, and function interfaces unchanged, and keep
Part A-only reporting functions only where they don't interfere with Part B.

### Prompt(s)
"I have completed and validated etl.py and features.py for Project A. For
Project B, copy their implementation into the corresponding files. Keep the
cleaning methodology, feature engineering, and function interfaces unchanged
unless the Project B brief requires a change. Do not introduce new cleaning
rules or alter existing logic. If a function is only needed for Part A
reporting, keep it only if it does not interfere with Project B; otherwise
explain why it can be removed. Then tell me what changed, why, and whether the
copied code is enough for portfolios.py, sentiment.py, and fusion.py."

### What the assistant produced
Copied both files verbatim (via `cp`, not retyped) and verified the code bodies
are byte-identical to Project A from the first import line onward. Made one
documentation-only edit per file: the module header no longer says "Project A",
and the `features.py` NOTE now points sentiment scoring to `src/sentiment.py`
instead of saying it "must not be added here". All Part A reporting functions
were kept, on the argument that none of them interfere with Part B (self-
contained, only run if called). Import-checked both modules against Project B's
`src/data_access.py` and confirmed the expected functions resolve. Reported that
the copy is a sound foundation but not sufficient on its own, since
`portfolios.py`, `sentiment.py`, and `fusion.py` are still `NotImplementedError`
stubs.

### What was wrong or risky
- Header still said "FINS3645 Project A"; the `features.py` NOTE ("must not be
  added here") was misleading in Part B, where scoring is in scope (it lives in
  `sentiment.py`).
- The assistant only import-checked the modules; it did not run the full
  pipeline 



## Entry 2 - Implement build_universe_returns in portfolios.py
### What I wanted
Implement only `build_universe_returns(universe: str) -> pd.DataFrame` in
`src/portfolios.py` - a wide daily-returns panel for "equity", "crypto", or
"combined". I explicitly forbade touching `etl.py`/`features.py` (validated,
frozen) and forbade reimplementing the equity/crypto calendar merge, since
`etl.merge_calendars()` already does it.

### Prompt(s)
"Read src/etl.py in full before writing anything - do not guess function
signatures. Note that etl.merge_calendars(equity_ret, crypto_ret) already
implements the equity/crypto calendar merge (left-join crypto onto the equity
trading calendar, EQ_*/CR_* prefixed columns) - do not reimplement this logic,
call it directly. Task: implement ONLY build_universe_returns(universe: str)
-> pd.DataFrame in src/portfolios.py. Do not implement anything else in this
file yet - no weight functions, no oos_backtest, no performance_metrics.
Spec: use etl.load_raw(), then etl.clean_equity() and/or etl.clean_crypto(),
then etl.compute_returns() on each cleaned panel separately (never merge price
levels before computing returns). For 'equity' and 'crypto': clean -> compute
returns -> pivot to wide (date index, one column per ticker, no prefix).
clean_crypto already caps the sample at 2023-12-31, don't re-cap. For
'combined': clean + compute_returns on both, pass both to
etl.merge_calendars() and return its output unchanged. Do not modify etl.py or
features.py; if either is missing something you need, stop and tell me."

### What the assistant produced
`build_universe_returns` in `src/portfolios.py`: validates the universe
argument (raises `ValueError` otherwise), loads the raw panels once via
`etl.load_raw()`, cleans + computes returns within each panel, pivots "equity"
and "crypto" to wide (bare ticker columns, `.sort_index()`), and returns
`etl.merge_calendars(equity_ret, crypto_ret)` unchanged for "combined". Left
the other two stubs (`oos_backtest`, `performance_metrics`) untouched as
instructed. The assistant then ran it against live data with the repo
interpreter and confirmed: equity 1006x50 (bare tickers), crypto 1461x10 (bare
tickers, capped at 2023-12-31 by clean_crypto), combined 1006x60 with EQ_/CR_
prefixes on the equity calendar, equity columns match the combined EQ_ columns
and the combined index equals the equity index, and the invalid-universe path
raises as expected. `ruff check` passed.

### What was wrong or risky
- The main correctness risk is the one the brief flags: computing returns
  across a merged calendar would fabricate returns at the seam between the
  equity and crypto calendars. The assistant avoided this by calling
  `compute_returns` within each panel before `merge_calendars`, and the
  combined panel inherits that guarantee from the frozen etl function rather
  than re-deriving it.
- `merge_calendars` returns a panel whose index has no name (`None`), unlike
  the equity/crypto pivots whose index is named `date`. This is inherited from
  the frozen etl function, so it is intentional, but it is a small
  inconsistency worth remembering when the backtest joins on dates later.
- No code defect found: I verified the function re-runs cleanly on live data
  and the shapes/prefixes/alignment are as specified.

### What I changed and why
Accepted the assistant's code as-is. Per my CLAUDE.md, when no error is found
I document what was checked rather than inventing a correction. Checks
performed: full-file read of `etl.py` first (signatures not guessed), real-data
run of all three universes plus the equity-vs-combined column/index alignment,
the invalid-input error path, and `ruff check`. No edits to `etl.py` or
`features.py`.


## Entry 3 - Baseline sentiment pipeline (VADER + FinVADER) and sector index
### What I wanted
A baseline Station 3 sentiment pipeline: score the Part A trading-day-aligned
headline panel with both VADER and FinVADER, aggregate to ticker-day then to an
equal-weight sector sentiment index, apply a one-trading-day lag to avoid
look-ahead, save the sector index (the required filename), and produce
comparison tables and figures (VADER vs FinVADER, sentiment over time, by
sector). Explicitly out of scope: no portfolio fusion, no touching portfolio
weights. I asked the assistant to explain assumptions and walk me through what
it did.

### Prompt(s)
"Implement the baseline sentiment pipeline for Project B using both VADER and
finVADER. Reuse the trading-day-aligned headline panel from Part A. Score each
headline, aggregate to ticker-day and then to a sector-level sentiment index,
applying a one-day lag to avoid look-ahead bias. Save the sector sentiment
index and generate comparison tables and figures (e.g., VADER vs finVADER,
sentiment over time, sentiment by sector). Do not implement portfolio fusion or
modify portfolio weights yet. Explain any assumptions made. Walk me through
what you did."

### What the assistant produced
`src/sentiment.py` with `score_headlines` (per-headline VADER and FinVADER
compound scores), `ticker_day_sentiment` (mean headline score per ticker-day,
reindexed to the full equity calendar with no-news days filled neutral),
`sector_sentiment_index` (equal-weight mean of the 5 sector tickers, then
`groupby(sector).shift(1)` lag), plus an end-to-end convenience wrapper; and
`scripts/sentiment_pipeline.py` that runs clean -> align -> score -> aggregate
-> lag and writes `results/data/sector_sentiment_index.csv`,
`results/data/ticker_day_sentiment.csv`, two comparison tables, and three
figures. It also added `finvader>=1.0.2` to requirements-dev.txt (build-only).

### What was wrong or risky
- Two real finVADER issues the assistant found and had to design around:
  (1) `finvader.finvader()` raises `UnboundLocalError` when both lexicons are
  off - its own defaults are broken, so both must be enabled; (2) the package
  rebuilds the VADER analyzer and ~7.4k-word lexicon on EVERY call (~10 min for
  147k headlines). The assistant replicated its exact lexicon merge once into a
  single analyzer and verified numerically that it matches
  `finvader.finvader()` on test headlines (all match).
- Look-ahead is the brief's highest-risk spot: the assistant lagged AFTER
  sector averaging. That is valid because an equal-weight mean commutes with a
  lag, but it is the kind of "clever" move that could hide a bug - I verified
  the lag by hand (Tech index on 2020-01-06 equals the mean of its tickers'
  scores on 2020-01-03, the prior trading day).
- Neutral-fill assumption needed scrutiny: no-news ticker-days score 0.0
  rather than being dropped or carried forward. Defensible (absence of news is
  its own state; carry-forward makes the index sticky) but it is a modelling
  choice the report must state.
- The data guide's prose says the sectors are "Comm/Telecom" and "Real Estate",
  but the actual data values are "Comm" and "RealEstate" - the assistant's
  first figure legend used the guide's names; I checked `load_equity_prices()`
  and corrected the ordering list to the real values. Also, the "sector" column
  printed in the summary table revealed the mismatch.
- I could not visually inspect the figures (my model can't render images in
  this environment), so I verified them by file size and dimensions and asked
  the student to eyeball them.

### What I changed and why
Accepted the pipeline as designed. Corrections made: (1) fixed the
`SECTOR_ORDER` list to the real sector strings ("Comm", "RealEstate") so the
summary table and by-sector figure sort deterministically; (2) cleaned up lint
(import ordering, long lines, unused noqa). Verification I performed: read
`etl.py`/`features.py` first to confirm the aligned-panel functions I asked to
reuse actually exist (`clean_news`, `build_text_panel`); re-ran the pipeline
end to end; confirmed 50x1006 ticker-day grid completeness; hand-checked the
lag and equal-weight math; confirmed VADER neutral share (48.9%) reproduces the
data guide's "about half neutral" warning and that FinVADER reclassifies 32.8%
of those neutrals; confirmed the required filename
`results/data/sector_sentiment_index.csv` is written. The results (r=0.60
between models; FinVADER moves 33% of VADER neutrals to a signed score) match
the expected VADER vs finance-lexicon story and are what I'll report, but I
will not claim the figures are visually clean until the student has looked at
them.


## Entry 4 - apply_sentiment() fusion rule in fusion.py
### What I wanted
A look-ahead-safe `apply_sentiment(weights, sentiment)` in `src/fusion.py` that
combines baseline portfolio weights with the one-day-lagged sentiment signal via
a transparent, reusable fusion rule. It must work with either VADER or FinVADER
through one code path, must not run the backtest or generate comparison outputs,
and - explicitly - I asked for the portfolio assumptions to be explained and
justified BEFORE any code was written.

### Prompt(s)
"Implement apply_sentiment() in fusion.py. It should combine the baseline
portfolio weights with the one-day-lagged sentiment signal using a transparent
and reusable fusion rule. Keep the implementation modular so it can be used with
either VADER or finVADER. Do not run the backtest or generate comparison outputs
here. Before coding, explain and justify any portfolio assumptions (e.g.,
long-only constraints, weight normalisation or other design choices) instead of
assuming them."

### What the assistant produced
A stated design first, then `apply_sentiment(weights, sentiment,
score_col="finvader_sentiment", tilt_strength=0.5)` implementing a
multiplicative tilt on a cross-sectional standardised signal:
`z = (s - mean(s)) / std(s)` over the equity subset, `w' = clip(w*(1 +
lambda*z), 0, inf)`, renormalised to sum to 1 per date. It pivots the long
ticker-day sentiment frame to wide, forward-fills onto the weight dates (most
recent sentiment on or before the rebalance date - never a backward fill), gives
crypto z = 0 (sentiment is equity-only), and falls back to baseline weights if a
date clips to all-zero. The `score_col` switch is how one rule runs either model.

### What was wrong or risky
- The core risk the brief names is look-ahead. The assistant moved the
  alignment inside the function (ffill reindex) rather than assuming the caller
  pre-aligns; the signal is already lagged by the pipeline, so it must never
  re-lag. I checked the alignment direction (forward, not backward) and that
  the contract is documented so a future caller can't accidentally pass raw
  unlagged sentiment.
- Standardising z is what makes lambda comparable across VADER (ticker-day std
  ~0.178) and FinVADER (~0.143); without it the "works with either model"
  requirement would be cosmetic. This is a genuine design choice, not a given -
  worth stating in the report.
- Degenerate cases needed explicit handling and were easy to get wrong: assets
  with no sentiment (crypto) must get z = 0 not NaN, a zero cross-sectional
  spread must not divide by zero, and an all-negative signal must not produce a
  zero-weight portfolio (falls back to baseline).
- Multiplicative vs additive tilt is a defensible-but-not-required choice; the
  student should confirm the tilt direction and lambda feel sensible before
  trusting a backtest result built on it.

### What I changed and why
Accepted the design and code as-is after verification, per my CLAUDE.md. Checks
I performed: unit-tested on the REAL `results/data/ticker_day_sentiment.csv`
artifact with synthetic equal-weight weights - rows sum to 1 and stay long-only;
`tilt_strength=0` returns the baseline exactly; crypto weights pass through
untouched; high-sentiment names gain relative weight and low-sentiment names
lose; both `score_col` values run; an all-negative signal falls back to
baseline. `ruff check` passes. No backtest was run and no comparison outputs
were generated, as instructed. Open item for the student: the tilt-strength
lambda (default 0.5) needs tuning in the backtest, and whether the tilt
actually adds value is the empirical question the backtest will answer - this
entry only established the mechanism.


## Entry 5 - Reusable transaction-cost model in portfolios.py
### What I wanted
A reusable transaction-cost model in `src/portfolios.py` as a portfolio
extension: 0.05% (5 bps) per rebalance charged on portfolio turnover, the
zero-transaction-cost baseline left unchanged, both gross and net portfolio
returns computed, with the turnover/cost maths explained up front. No final
figures or comparison tables.

### Prompt(s)
"Implement a reusable transaction-cost model in portfolios.py as a portfolio
extension. Assume a transaction cost of 0.05% (5 basis points) per rebalance
based on portfolio turnover. Keep the original zero-transaction-cost baseline
unchanged, calculate both gross and net portfolio returns, and explain how
turnover and transaction costs are computed. Do not generate final figures or
comparison tables here."

### What the assistant produced
Three reusable functions appended to `portfolios.py` (the existing
`oos_backtest`/`performance_metrics` untouched): `portfolio_turnover(weights,
include_initial=False)` computing one-way turnover = 0.5 * sum_i |w_i(t) -
w_i(t-1)| (nonzero only on rebalance dates, since weights are flat between
them); `transaction_cost(turnover, cost_rate=0.0005)` = turnover * rate; and
`apply_transaction_costs(gross_returns, weights, cost_rate, include_initial)`
which layers net = (1 + gross) * (1 - cost) - 1 on rebalance days onto the
existing gross series and returns a frame [gross_return, turnover, cost,
net_return], validating date alignment between weights and returns. The module
docstring records the cost assumptions (5 bps, one-way turnover, cost paid at
the start of the rebalance day, initial purchase excluded so funds are compared
for rebalancing only).

### What was wrong or risky
- The turnover convention is easy to get wrong and changes costs by 2x: the
  correct one-way definition for a long-only, fully invested fund is HALF the
  gross absolute weight change (selling X to buy X trades X, not 2X). I
  checked the 0.5 factor and the docstring states the convention so the report
  can cite it.
- Where the cost lands on the daily series is a look-ahead-adjacent subtlety:
  the cost is paid at the START of the rebalance day, so the day's return must
  accrue on the post-cost value - net = (1+gross)(1-cost)-1, not gross - cost
  (which would be the same only for tiny values but is wrong in general). The
  assistant verified this formula numerically.
- First-date treatment: the cash-to-weights purchase is excluded by default
  (include_initial=False) so every fund is charged for the same thing; the
  alternative is available as a parameter. A silent wrong default here would
  bias every fund's net numbers.
- The assistant found the file had three pre-existing ruff failures (import
  sort, two long lines) and fixed them as formatting-only changes, then
  re-verified the backtest behaviour was unchanged.

### What I changed and why
Accepted as-is after verification. Checks I performed: synthetic unit checks
(first row turnover 0, turnover = 0.2 for a 0.4 gross weight move, cost =
0.0001, include_initial gives 0.5, net formula, non-rebalance days net =
gross, gross column bit-for-bit identical to the input baseline); a real
integration run on `combined min_variance` (250 OOS days, 11 traded rebalance
dates, mean one-way turnover ~1.2%, total drag ~0.7 bps for the year - a
sensible magnitude for a low-turnover method, which is why gross vs net look
equal at 4 dp); and ruff-clean after the formatting fixes. No figures or
comparison tables generated, as instructed. Note for the student: turnover
realism (5 bps flat) is a stated assumption and the numbers will differ if
they change it, and the cost model only matters once fused/tilted funds - which
turn over more - are backtested.


## Entry 6 - Long-only minimum-CVaR portfolio method in portfolios.py
### What I wanted
Add a long-only CVaR portfolio method to `src/portfolios.py` as an extension to
the existing baseline methods, using only past training-window returns and
compatible with the existing walk-forward `oos_backtest`. I explicitly required
the CVaR confidence level, objective, and constraints to be explained BEFORE any
code was written, and forbade replacing or removing the baseline methods.

### Prompt(s)
"Add a long-only CVaR portfolio method to portfolios.py as an extension to the
existing baseline methods. Use only past training-window returns at each
rebalance and keep it compatible with the existing walk-forward backtest.
Explain the CVaR confidence level, objective and constraints before coding. Do
not replace or remove the baseline methods."

### What the assistant produced
Stated the design first, then implemented `cvar_min(window, alpha=0.95,
max_weight=1.0) -> pd.Series` and registered it in the `METHODS` dict under
"cvar_min" (the existing four methods untouched). Design: minimise CVaR_alpha
of daily portfolio LOSSES at confidence alpha=0.95 (expected loss in the worst
5% of training-window days) via the exact Rockafellar-Uryasev (2000) LP,
solved with `scipy.optimize.linprog` method="highs", subject to fully invested
(1'w=1) and long-only (0 <= w <= max_weight). At the optimum the auxiliary
beta variable is the VaR of the loss distribution and the objective equals the
CVaR. It drops incomplete rows (`dropna(axis=0)`) inside the window before
building the scenario matrix - `_clean_window` keeps assets whose leading NaN
is before their first trade, and an LP scenario needs a complete
portfolio-return row. Added a module-docstring bullet recording the CVaR
assumption for the report.

### What was wrong or risky
- First implementation passed the window to `linprog` with NaN rows intact,
  which crashed ("A_ub must not contain values inf, nan") because
  `_clean_window` legitimately tolerates leading NaNs before an asset's first
  trade. Fixed by dropping incomplete rows before building the LP; on the real
  combined universe this only removes the handful of leading equity/crypto
  calendar-seam days (756 rows -> 755 complete).
- The LP is the one weight method that is exact and convex (highs interior
  point), unlike the SLSQP optimisers, so the "silent stall" failure mode the
  brief warns about is structurally impossible here - worth stating, since the
  other methods needed their own stall guards.
- The tail-loss objective is not obviously different from min_variance's until
  verified, so I sanity-checked weights genuinely differ across methods and
  that the LP's optimum is real (brute-force tail-average check).
- A no-look-ahead check on non-rebalance dates initially showed a mismatch,
  which was my test's bug, not the code's: weights are held FLAT between
  rebalances, so only rebalance dates (plus the forced first OOS date) should
  reproduce the recomputed weights. Restricting the check to rebalance dates
  gives max diff 0.0.

### What I changed and why
Accepted the code after verification, per my CLAUDE.md. Checks I performed:
`ruff check` on portfolios.py (clean; the remaining repo lint errors are all in
the frozen etl.py/features.py/data_access.py, untouched); full test suite
(2 passed); real-data sanity check on the combined universe (weights sum to 1,
all >= 0, 13/60 names held, L1 distance from equal_weight 1.61 and from
min_variance 0.59 - genuinely different); independent brute-force CVaR check
confirmed the LP solution is the true minimum (0.0243 tail loss vs 0.0433 for
equal weight and 0.0249 for min_variance); full `oos_backtest(combined,
"cvar_min")` runs (250 OOS days, 11 traded rebalances, ann_ret 8.9%, vol
11.6%, Sharpe 0.77, maxDD -7.2%); and exact reproduction of the backtest
weights by recomputing `cvar_min` on strictly-past windows at every rebalance
date (max diff 0.0) - no look-ahead. Open item for the student: whether
min-CVaR adds value over min_variance on this data is an empirical question;
with alpha as a parameter they can also test 90%/99% sensitivity for the
report.


## Entry 7 - Coverage-weighted sentiment index in sentiment.py
### What I wanted
Add a third sentiment index as an additional option in `src/sentiment.py`: a
coverage-weighted sector index using the NUMBER OF HEADLINES per ticker-day as
a confidence measure, while leaving the original equal-weight VADER and
FinVADER indices byte-for-byte unchanged. I required the weighting approach to
be explained BEFORE implementation and all three indices saved for later use in
fusion.py.

### Prompt(s)
"Extend sentiment.py by adding a coverage-weighted sentiment index as an
additional option. Keep the original VADER and finVADER indices unchanged. Use
the number of headlines available for each ticker-day as a confidence measure
when constructing the new sentiment index, explain the weighting approach
before implementation, and save all three indices for later use in fusion.py."

### What the assistant produced
Explained the design first, then added two functions to `sentiment.py`
(existing ones untouched): `ticker_day_coverage(panel, calendar_dates)` -
headline count per ticker-day reindexed onto the full equity calendar (0 for
no-news days) - and `coverage_weighted_sector_sentiment_index(ticker_day,
coverage, sector_map, lag=1)` which computes index(s,t) = sum_i n(i,t)*score /
sum_i n(i,t) per sector-date, then lags one day exactly like the equal-weight
index. The pipeline script now saves the third artifact
`results/data/coverage_weighted_sentiment_index.csv` alongside the unchanged
`sector_sentiment_index.csv`; both carry the same two model columns so fusion's
score_col switch can consume either.

### What was wrong or risky
- The weighting choice changes the no-news treatment and that is the point to
  get right: zero-headline ticker-days get weight 0 (excluded), unlike the
  equal-weight index which counts them as neutral 0.0 with full weight.
  Consequence: a sector-day with NO headlines at all has no reading (0/0 ->
  NaN, dropped after the lag) - 228 of 10060 sector-days on real data,
  concentrated in thinly-covered sectors (Materials 66, RealEstate 66,
  Utilities 44, the rest spread across the other seven), and spread across the
  whole sample (2020-06-10 to 2023-12-29; only 76 in 2023). Not a signal of
  neutrality - an absent signal - so it is dropped rather than asserted as
  0.0. In fusion, apply_sentiment's forward-fill would hold the last
  trustworthy reading instead of a false neutral - a deliberate property,
  stated in the module docstring.
- The lag still commutes with the aggregation (per-sector date shift after the
  weighted mean), so the no-look-ahead guarantee is preserved; I verified the
  weighted mean against a manual group-by computation and the lag by
  reconstructing a value from the strictly-prior day.
- The coverage-weighted index is only modestly correlated with equal-weight
  (VADER r=0.76, FinVADER r=0.80) and differs in row count (9831 vs 10050
  lagged rows), so it is a genuinely different signal, not a cosmetic
  re-weighting - worth stating in the report.

### What I changed and why
Accepted the code after verification, per my CLAUDE.md. Checks I performed:
`ruff check` on both files (clean); full pipeline re-run (scored 146,836
headlines, 50x1006 ticker-day grid, third index saved); independent hand-check
of the weighted mean on a real Tech sector-day (manual group-by matches the
function to 1e-12 for both models); lag check (value on day t equals the
pre-lag previous day's value); zero-headline NaN drop confirmed (228 sector-
days); correlation vs equal-weight confirms the indices genuinely differ; all
three artifacts load with the expected schema; test suite still 2/2. Note for
the student: fusion.py currently consumes ticker-day sentiment, not the sector
index - wiring the coverage-weighted index into a fusion backtest is a
separate step, and the "hold last reading" behaviour during news gaps is a
modelling choice they should confirm for the report.


## Entry 8 - Full consistency review of Project B codebase (read-only)
### What I wanted
A read-only consistency assessment of the whole Part B codebase against
`PROJECT_BRIEF.md`, `context/`, and the rubric: data schemas, function I/O, date
alignment, no-look-ahead, annualisation, portfolio assumptions, sentiment
lagging, and whether outputs can connect through `scripts/run_part_b.py`.
Explicitly no file changes.

### Prompt(s)
"Perform a read-only consistency assessment of the Project B codebase. Read the
brief, AGENTS.md, context/, and src/data_access.py, src/etl.py, src/features.py,
src/portfolios.py, src/sentiment.py, src/fusion.py before assessing. Report
confirmed-correct components, bugs/inconsistencies/missing dependencies,
out-of-scope or duplicated features, missing functionality required before
run_part_b.py can work, embedded modelling assumptions, and a keep/revise/remove
recommendation per item. Do not modify any files."

### What the assistant produced
A sectioned report covering: confirmed-correct components (etl cleaning rules,
`build_universe_returns` shapes/prefixes, `oos_backtest` monthly expanding-window
mechanics, all five weight methods being genuinely different and summing to 1,
transaction-cost maths, sentiment scoring and lag, coverage-weighted index);
bugs/inconsistencies (fusion's lag contract is not met by the unlagged
`ticker_day_sentiment.csv` artifact -> latent look-ahead; fusion silently no-ops
on combined `EQ_`/`CR_` columns; `run_part_b.py` is a stub; three required
artifacts missing; `vaderSentiment` undeclared in requirements-dev);
out-of-scope items (coverage index, `cvar_min`, transaction costs are
innovations, not brief requirements); missing functionality before the runner
can work; embedded modelling assumptions (long-only, rf=0, monthly rebalance,
252/365 annualisation, neutral fill, tilt rule); and a keep/revise/remove table.

### What was wrong or risky
- The two real code defects were the fusion lag contract (would create
  look-ahead the moment it is wired) and the silent combined-fund no-op. Both
  are fixed in Entries 9-10.
- The biggest gap: `run_part_b.py` - the brief's required entry point - is a
  stub, so three of four required outputs and the whole app layer do not exist
  yet.
- All four optimisers enforce long-only (bounds `(0,1)` for min_variance and
  max_sharpe, `(1e-8,None)` for the risk-parity log-barrier, `(0,max_weight)`
  for cvar_min) plus a post-hoc `np.clip`. Unconstrained max_sharpe would
  short. Decided to KEEP long-only: the brief frames funds as investable
  products, and removing the constraint would break risk-parity's
  well-posedness and the transaction-cost/fusion semantics that assume
  w >= 0.

### What I changed and why
No code changed (read-only, per instruction). Recorded the keep/revise/remove
conclusions so the follow-up fixes are deliberate: revise the fusion lag
contract, revise the prefix no-op, revise `run_part_b.py`, revise
`streamlit_app.py`, add `vaderSentiment` to requirements-dev, keep cvar_min /
coverage index / transaction costs as claimed innovations. Also flagged the
student decision list (combined-fund annualisation statement, tilt strength,
fund grid, which innovations to actually run) for a later entry.


## Entry 9 - Fix sentiment look-ahead bias in fusion.py
### What I wanted
Fix the latent look-ahead risk in `apply_sentiment()`: its contract said the
input "must already be lagged one trading day", but the pipeline writes
`ticker_day_sentiment.csv` UNLAGGED (the one-day lag lives only inside the
sector-index functions). Wiring fusion with that artifact would tilt day-t
weights on day-t news. The fix should make fusion safe no matter what it is
given, without breaking the `weights`/`sentiment` interface.

### Prompt(s)
"Fix the sentiment look-ahead bias in fusion.py. apply_sentiment currently
assumes the ticker-day sentiment input is already lagged, but the artifact it
names is written unlagged by the pipeline. Make the function apply the one-
trading-day lag itself so fusion is always safe. Keep the weights/sentiment
interface and the rest of the tilt rule unchanged. Explain the new contract
before coding."

### What the assistant produced
The lag now happens inside `apply_sentiment()`: after pivoting the long
ticker-day frame to wide, the panel is shifted by one row (`shift(1)`) before
the forward-fill alignment onto the weight dates. Shift is on the sentiment
frame's own trading calendar, so it is a per-ticker one-trading-day lag; the
existing `reindex(weights.index, method="ffill")` then picks the latest value ON
OR BEFORE each rebalance date. Combined: the tilt at date t uses news from t-1
or earlier.

### What was wrong or risky
- The original design deliberately did NOT re-lag, on the assumption the caller
  pre-lagged. That assumption is what was wrong: the pipeline never lags the
  ticker-day artifact, so the contract was unsafe by construction.
- A caller passing an already-lagged signal now gets a conservative extra day
  (never look-ahead), and passing the sector-level index fails loudly (no
  `ticker` column for the pivot) rather than silently - both acceptable.

### What I changed and why
Updated the module docstring and the `sentiment` parameter contract: the input
is now RAW ticker-day sentiment (same-day news) and `apply_sentiment` applies
the one-trading-day lag internally before alignment. Verified with the real
artifact: a date-t tilt uses only the prior trading day's sentiment, and the
first OOS rebalance date picks up the pre-2023 reading. `ruff check` clean.


## Entry 10 - Fix silent no-op in fusion.py for combined funds (EQ_/CR_ mapping)
### What I wanted
Fix the silent no-op in `apply_sentiment()`: combined-fund weights have
`EQ_AAPL` / `CR_BTC-USD` style columns, but the sentiment lookup matched on bare
ticker names, so `tiltable` was empty and the tilt returned baseline weights
with no warning. Map the `EQ_`/`CR_` prefixes back to the underlying ticker so
the equity leg of a combined fund is actually tilted, with crypto left at z=0.

### Prompt(s)
"Fix the silent no-op in fusion.py. apply_sentiment matches weight columns
against bare ticker sentiment columns, so combined-fund weights (EQ_/CR_
prefixes) find no match and the tilt silently does nothing. Map EQ_/CR_ prefixed
columns back to the original ticker names for the sentiment lookup, keep crypto
(CR_) at z=0, and keep bare equity tickers working unchanged."

### What the assistant produced
`apply_sentiment` now builds an explicit weight-column -> sentiment-ticker map:
a column is tiltable if it is itself a sentiment ticker (equity-only funds,
unchanged path), or if it starts with `EQ_` and the stripped name is a sentiment
ticker (combined funds). `CR_` columns and anything else are never mapped, so
crypto still gets z=0 and passes through untouched. The cross-sectional z is
computed on the mapped sentiment columns and written back into the weight
columns, so the tilt now reaches the equity leg of a combined fund.

### What was wrong or risky
- Before the fix the function could not be trusted on combined weights: it
  returned the baseline silently, which would have made a fused-combined
  backtest look like "sentiment adds nothing" when it was actually running
  nowhere.
- The fix must not change the equity-only behaviour (identity mapping) - I
  verified that path is byte-for-byte the same numbers as before.

### What I changed and why
Rewrote the tiltable-column selection and z-computation to use the map; updated
the module docstring to state that combined weights are handled by stripping the
`EQ_` prefix and that crypto stays z=0. Verified: the identity mapping leaves the
equity-only path structurally unchanged (its numbers differ from pre-Entry-9
behaviour only by the intended one-day lag); combined weights now produce a
tilted EQ_ leg whose z matches a manual lagged z over the bare names, CR_ columns
pass through with their relative proportions intact; rows still sum to 1 and stay
long-only; `ruff check` clean.


## Entry 11 - Build the run_part_b.py reporting pipeline (all Station 3 artifacts)
### What I wanted
A single entry point (`scripts/run_part_b.py`) that reproduces every Station 3
report/app artifact: 12 baseline funds (3 families x 4 required methods), 3
CVaR extension funds, 4 sentiment-fused funds, transaction-cost layering, and
the required CSVs/figures - while keeping CVaR separate from "baseline"
terminology, recomputing fused returns from fused weights, and staying
look-ahead safe.

### Prompt(s)
"Implement run_part_b.py as the full Station 3 reporting pipeline with these
final decisions: 15 baseline+CVaR funds, 4 fused funds (Equity/Combined
Min-Variance x VADER/FinVADER), fused returns recomputed as
(fused_weights * returns).sum(1), artifact schemas fund_returns
[fund,date,gross_return,net_return], fund_weights [fund,date,ticker,weight],
performance_metrics [fund,basis,n_days,ann_return,ann_vol,sharpe,max_drawdown,
mean_turnover,total_cost] with basis in {gross,net}, lambda=0.5 fixed, no app
work, verify outputs."

### What the assistant produced
A ~570-line runner that (1) regenerates the sentiment artifacts by reusing the
verified functions from scripts/sentiment_pipeline.py (loaded via
importlib so its save_outputs/figures are shared, not duplicated); (2) runs
oos_backtest per family x method and per fused fund (fused weights tilted at
rebalance dates via fusion.apply_sentiment, held flat, returns recomputed);
(3) layers 5 bps one-way transaction costs on every fund; (4) writes
results/data/fund_returns.csv, fund_weights.csv, sector_sentiment_index.csv,
results/tables/performance_metrics.csv (plus fusion_before_after.csv and
fusion_lambda_sensitivity.csv), and 6 new figures; (5) prints a verification
report (fund counts, weight-row sums, gross-vs-net, method diversity, tilt
activity).

### What was wrong or risky
- The first fused rebalance is 2023-01-03; without proof the tilt uses only
  2022 sentiment, the whole fusion is suspect. I initially checked this by
  comparing index labels, which was wrong (shift is positional, not
  date-aligned). The correct check confirmed lagged[2023-01-03] == raw
  [2022-12-30] elementwise, and that raw 2023-01-03 sentiment is nonzero (the
  lag is doing real work, not silently inactive).
- Verified manually: a gross return on 2023-01-03 recomputed from weights x
  returns matches oos_backtest to <1e-12; a net return on a rebalance day
  matches (1+gross)(1-cost)-1; every fund's weights sum to 1 on every day;
  pairwise mean-weight L1 distances across methods are >= 0.12 in all three
  families (methods genuinely differ, no silent optimizer stall).

### What I changed and why
Added a `fund_type` column (baseline/extension/fusion) to performance_metrics.csv
beyond the agreed schema so the CVaR funds are explicitly not labelled
baseline. A genuine analytical finding surfaced during verification:
Combined Minimum Variance allocates exactly 0 to crypto in every window (crypto
volatility >> equity, so min-variance excludes it), making the combined
min-variance fund and its two fused variants effectively identical to the
equity ones (returns differ only at ~1e-8). This is a real result to
acknowledge in the report, not a bug. Also noted: crypto funds show very high
2023 net Sharpe (~3.5-3.6), consistent with the strong 2023 crypto rally, and
should be framed as a period effect. Ruff clean; full pipeline re-run
verified after the fix. No app changes (deferred until report results are
reviewed).

---

## Entry 12 - Convert every figure to FT style, split crowded ones
### What I wanted
Apply the Project A FT house style to ALL report figures, not just growth of
$1. Where a single PNG would get crowded or overlap, split it into separate
PNGs (the user will put them side by side to compare), keeping the file names
consistent for similar figures (same base name, different family/model suffix).

### Prompt(s)
"apply to all the figures with FT style, if one png cannot fit all the
information and lead to overlap, just seperate them i can put them together to
compare, that totolly, just ensure that the name of the figure is identical for
similar figures."

### What the assistant produced
Converted all six runner figures and all three sentiment-pipeline figures to
FT style (src.plotting, reused from Project A) and split every crowded one,
with consistent `<type>_<family>` naming:
- growth_of_dollar_{equity,crypto,combined}.png
- drawdown_{equity,crypto,combined}.png
- weights_over_time_combined_{min_variance,max_sharpe}.png
- sharpe_by_fund_{equity,crypto,combined}.png (horizontal grouped gross-vs-net
  bars, from the crowded 19-fund single chart)
- fusion_before_after_{equity,combined}.png
- turnover_cost_{equity,crypto,combined}.png (bars coloured by method)
- sentiment_sector_index_over_time_{vader,finvader}.png (10 sector lines each,
  borderless FT legend)
- sentiment_vader_vs_finvader.png and sentiment_by_sector.png (FT frame, kept
  as single figures - not crowded)

Drawdown uses the helper's borderless-legend fallback (all five lines end near
0, so end-labelling would overlap); weights area charts moved their legend
outside the plot; the stale multi-panel PNGs were deleted.

### What was wrong or risky
- Two subtle traps when reusing Project A's helpers: `ft_grouped_bar` already
  turns OFF the y-axis grid and ON the x-axis grid, so the ft_figure() default
  (y grid) must not be re-applied; and `ft_multi_line_end_labels` falls back
  to a legend when there are more than `max_end_labels` series, which is the
  right behaviour for drawdown but not for growth (5 end-labels fit).
- This model cannot view images, so visual overlap can only be judged by the
  user; the split-into-more-PNGs decision is a deliberate hedge against
  crowding.

### What I changed and why
Rewrote fig_drawdown/weights/sharpe/fusion/turnover_cost in run_part_b.py and
fig_over_time/scatter/by_sector in sentiment_pipeline.py to use src.plotting,
and split the crowded ones so each PNG carries one family/method. Naming is
identical for similar figures (same stem, family suffix), which the user
confirmed is how they will compare them. Ruff clean; full pipeline re-run
after the change reproduces identical metrics and all 20 figures render
(verified sizes/dimensions); check_handin.py still 21/21.

---

## Entry 13 - Fix legend-over-line overlap on drawdown (and other legend-position) figures
### What I wanted
The user reported the legend overlaps the lines on every drawdown figure. Fix
the overlap without losing the FT style, and make sure the same class of
problem (a legend sitting inside the plot over the data) is not present on
the other figures.

### Prompt(s)
User report: "the legend is overlap with the lone" for the drawdown figures.
Follow-up: keep FT style, keep the CVaR dashed line, verify the fix.

### What the assistant produced
The drawdown figures placed a borderless FT legend in the top-left of the
axes (the `ft_multi_line_end_labels` fallback). That box covered the data band
drawdown ~0.000..-0.028 (verified: legend covered data y range -0.0281..+0.0039
at the top of the axes), exactly where all five lines start near 0 and recover
near 0, so the overlap was real.

### What was wrong or risky
- A naive overlap test that only counted sampled data points inside the legend
  box reported "no overlap" (first-day points sit a hair above the legend's top
  edge), which shows point-sampling is the wrong way to judge this; the box
  clearly intersected the lines' home band.
- `Legend.set_bbox_to_anchor()` on the Sharpe figure was not enough on its own:
  it keeps the legend's original `loc` ("lower right"), so only the right edge
  moves out and the box still overlaps the bars. The legend must be re-created
  with `loc="center left"` + `bbox_to_anchor=(1.01, 0.5)`.

### What I changed and why
- fig_drawdown (run_part_b.py): re-create the legend with `loc="upper center"`,
  `bbox_to_anchor=(0.5, -0.14)`, `ncol=5` so it sits BELOW the chart, and save
  with `bbox_inches="tight"` so it is not cropped.
- fig_sharpe (run_part_b.py): re-create the Gross/Net legend with
  `loc="center left"`, `bbox_to_anchor=(1.01, 0.5)` (outside the right edge)
  and `bbox_inches="tight"`.
- fig_weights (run_part_b.py) and fig_by_sector, fig_over_time
  (sentiment_pipeline.py): same outside-the-plot treatment for their legends.
- Verification: a throwaway script that replays the exact shipped figure bodies
  on the saved artifacts and asserts the legend box is entirely below
  (drawdown, sentiment over-time) or to the right of (sharpe, weights,
  by-sector) the axes box. After the fix all 11 checked figures pass with
  zero legend/axes overlap; metrics byte-identical to Entry 11/12; 21/21
  hand-in checks still pass.

---

## Entry 14 - Baseline-only Sharpe figures (gross, five methods, one chart per family)
### What I wanted
Regenerate the Sharpe ratio charts so they show ONLY the baseline performance:
the five optimisation methods (Equal Weight, Minimum Variance, Maximum
Sharpe, Risk Parity, Minimum CVaR) measured on gross returns only, one chart
per asset universe (Equity, Crypto, Combined), same 2023 out-of-sample period
and FT styling. Exclude VADER/finVADER and fused portfolios, gross-vs-net
comparisons, and transaction costs/turnover - those belong to the innovation
sections later in the report. Replace the existing sharpe_by_fund figures.

### Prompt(s)
"Produce the baseline performance figures only. Regenerate the Sharpe ratio
charts using gross returns only and compare only the five optimisation
methods ... Create one chart for each asset universe ... Do not include: VADER
or finVADER portfolios, Fused portfolios, Gross vs net comparisons,
Transaction costs or turnover ... Do not change any backtest logic or
portfolio calculations - only regenerate the baseline Sharpe figures. Replace
the share_by_fund figures."

### What the assistant produced
Rewrote fig_sharpe in run_part_b.py to plot a single gross-only horizontal
bar per fund (coloured by method via the existing COLORS map), keeping the
same sharpe_by_fund_{equity,crypto,combined}.png filenames. The five methods
appear in the user's requested order (Equal Weight, Minimum Variance,
Maximum Sharpe, Risk Parity, Minimum CVaR), matching the funds-list order the
runner already uses. No legend is needed for a single-series chart.

### What was wrong or risky
- The previous version was a grouped Gross-vs-Net chart with a legend outside
  the plot - exactly what the user now wants removed. A naive edit that only
  dropped the Net series would have left the unused legend and the "gross vs
  net" title in place.
- Reusing ft_bar (which does not set explicit y ticks) followed by
  set_yticklabels triggers a benign matplotlib UserWarning; fixed by calling
  set_yticks(range(len(names))) first.

### What I changed and why
Single gross-only bar per fund, methods in the fixed order, method-coloured
bars, x-label "Sharpe ratio (gross returns, rf = 0)" and title "...Sharpe,
2023 (gross)". No backtest or portfolio code touched. Verified programmatically
that each regenerated PNG has exactly 5 bars with no legend and gross-only
values matching performance_metrics.csv (e.g. Equity Maximum Sharpe 2.0486,
Crypto Minimum Variance 3.6022); the Combined Minimum Variance/CVaR bars equal
the Equity ones (0.4907/0.7656) because combined min-variance allocates 0 to
crypto - the documented finding, not a bug. Full pipeline re-run reproduces
identical metrics; legend-geometry check updated (sharpe now expects no
legend) and all figures pass; ruff clean; check_handin.py 21/21.

---

## Entry 15 - Publication 2x5 sector sentiment grid (FinVADER, 21-day average)
### What I wanted
A publication-quality report figure showing how sentiment evolves over time
across every equity sector, rather than comparing portfolio performance: a 2x5
grid with one subplot per sector, the same y-axis limits on every panel, a
zero-sentiment line per panel, years on the x-axis, and the report palette
(dark burgundy line on the light beige background).

### Prompt(s)
"Create a publication-quality figure for the report titled: Sector sentiment
over time (FinVADER, lagged index, 2020-2023)... one subplot for each equity
sector (Communication Services, Consumer, Energy, Financials, Healthcare,
Industrials, Materials, Real Estate, Technology, Utilities). Arrange the
subplots in a 2x5 grid... same y-axis limits... horizontal line at sentiment
= 0... dark burgundy line, light beige background. Title: Sector sentiment
over time (FinVADER, 21-day average, 2020-2023)... Label years on the x-axis."
Follow-ups: keep the y-axis labels on the second row too, then narrow the
vertical gap between the two rows.

### What the assistant produced
Added fig_sector_grid_finvader to scripts/sentiment_pipeline.py and wired it
into save_outputs, producing results/figures/sentiment_sector_grid_finvader.png.
One panel per sector in the requested order (2 rows x 5 columns), symmetric
shared y-limits, a zero line per panel, explicit 2020-2023 year ticks on the
bottom row, x tick labels suppressed on the top row, y tick labels on both
left-column panels, FT background/spines, and the en-dash title as requested.

### What was wrong or risky
- The required title says "21-day average" but the pipeline artifact is the
  daily lagged index. Plotting the raw daily series under that title would be
  mislabelled, so the figure plots the 21-day rolling mean of the lagged
  index and states this on the y-axis.
- First passes suppressed axis labels on interior panels: the top row showed
  no x-axis (years) and the first row beyond Communication Services showed no
  y-axis. The user required every line chart to be a complete chart, so each
  of the 10 subplots must carry its own y-axis and x-axis (years).
- The date column reads back from CSV as strings, so YearLocator ticked the
  wrong decades (1969-1972) until dates were coerced with pd.to_datetime, and
  margin ticks (2019/2024) leaked in until ticks were set explicitly.
- The default hspace left a large blank band between the rows; the second-row
  titles then floated far from their panels.

### What I changed and why
Drew the 21-day rolling average (min_periods=1) of the lagged FinVADER index
and labelled it accurately; removed both x- and y-axis label suppression so
every one of the 10 panels is a complete chart with its own y-axis and year
ticks (2020-2023); coerced dates to datetime and set explicit year ticks
(2020-2023); reduced hspace from 0.42 to 0.24. Verified programmatically that
all 10 panels share one symmetric ylim, every panel has a zero line and
burgundy (#7B1E3D) line on beige (#F7F1E8), the years read 2020-2023, every
panel shows its own x- and y-axis, no labels collide between neighbouring
columns, and the row-1 year labels still clear the row-2 panels and titles in
the narrowed gap. Full pipeline re-run regenerated the PNG; ruff clean;
check_handin.py 21/21.

---
## Entry 16 - Sentiment tilt on portfolio weights (horizontal grouped bars)
### What I wanted
A publication-quality figure placed before the performance results showing the
fusion mechanism rather than another performance comparison: for the Equity
Minimum Variance baseline and its + VADER / + finVADER variants, the average
monthly weight change delta w = w_fused - w_baseline per stock across all 2023
out-of-sample rebalances. Horizontal grouped bar chart, VADER and finVADER bars
side by side per stock, bars straddling a zero reference line, the average
weight change printed at the end of each bar (no overlaps), x-axis "Average
Weight Change (%)", y-axis stock ticker, clear legend, and the title "Average
Weight Changes after Sentiment Fusion (Relative to the Baseline Minimum
Variance Portfolio)".

### Prompt(s)
Provided as the task brief: compute per-stock monthly weight differences for
both lexicons, average across all out-of-sample rebalances, select the 10
stocks with the largest absolute average change, and plot as a horizontal
grouped bar chart around zero with legend, zero line, FT formatting and the
specified title. Follow-up: rename the title to "Average Weight Changes after
Sentiment Fusion (Relative to the Baseline Minimum Variance Portfolio)" and
print the average weight change at the end of each bar without overlapping
labels.

### What the assistant produced
fig_fusion_weight_tilt in scripts/run_part_b.py wired into main(); writes
results/figures/fusion_weight_tilt_equity.png. Samples the daily
(monthly-rebalanced, ffill) weights of the three funds at each of the 12
monthly OOS rebalance dates (first trading day of the month, 2023-01-03 to
2023-12-01), averages the per-stock deltas across rebalances, ranks by
max(|mean VADER delta|, |mean finVADER delta|), and draws the top 10 as
horizontal grouped bars around a zero line in the FT palette (VADER blue
#0F5499, finVADER teal #1E7A6E). Every bar is labelled at its end with the
signed average weight change ("+2.93", "-2.59", ...), placed outside the bar
so labels never overlap. Stocks are sorted by absolute weight change with the
largest tilt at the top (WMT, NEM, EA, MRK, MMM, ABBV, PSA, GILD, T, UPS).

### What was wrong or risky
- The long figure title (93 chars at fontsize 11 bold) overflowed the figure
  width by ~10% (1905 px vs 1720 px), so it would have been clipped at the
  right edge - checked programmatically before trusting the export.
- The below-axes legend could be clipped by tight_layout, so the export needed
  bbox_inches="tight" as well as tight_layout.
- The renamed title is almost as long as the original, so it needed wrapping
  to the same two-line layout; and un-guarded value labels would collide for
  bars with similar magnitudes, so each label is offset beyond its own bar end
  (left-aligned after positive bars, right-aligned before negative bars) and
  the horizontal padding was widened.
- The first pass sorted the selected 10 stocks ascending by tilt magnitude and
  then inverted the y-axis, which put the SMALLEST tilts at the top (UPS, T)
  and the largest at the bottom - the opposite of the intended reading order.

### What I changed and why
Renamed the title to the requested wording, kept it bold/left-aligned and
wrapped over two lines so it fits, and added the average weight change at the
end of every bar (signed, two decimals) with offsets beyond each bar end so no
labels overlap; saved with bbox_inches="tight" so the legend below the chart is
kept. The figure uses the same rebalance dates and convention as the backtest
itself (no prepended first-OOS date if it is already the month's first trading
day), so the tilt shown is the tilt actually traded. Verified independently from
fund_weights.csv: 12 rebalances, top-10 tickers WMT/NEM/EA/MRK/MMM/ABBV/PSA/
GILD/T/UPS, mean deltas sum to 0 for both lexicons (weights stay normalized),
the largest tilts are WMT (+2.9pp VADER / +3.6pp finVADER) and NEM
(-2.6pp / -2.3pp), the title fits the figure width, all 20 value labels match
the CSV-derived deltas exactly, no two labels overlap, and the rows render
largest-tilt-first top to bottom. Full pipeline re-run regenerates the PNG;
ruff clean; check_handin.py 21/21.

---

## Entry 17 - Build the EverVest Streamlit app (Home-first, results-only)
### What I wanted
Create the Station 4 Streamlit app from scratch, starting with the Home page
so I can see it locally before the remaining pages are finalised (we can
adjust anytime). I named the product **EverVest** and supplied the value
proposition verbatim for the hero: "EverVest's value proposition is therefore
to help long-term retail investors compare diversified, systematic investment
opportunities by integrating historical returns, downside risk and company
news into a single evidence-based decision-support platform." Hard
constraints from the brief/AGENTS.md: the app loads ONLY precomputed
artifacts from `results/` (no rerunning backtests or VADER), must never import
`data_access`, nltk, or finvader (cold-start constraint, app deps only), must
preserve the four required filenames, and must keep the FT house style I built
for the report. I wanted a product, not a generic analytics dashboard.

### Prompt(s)
User brief: "build the evervest app" with the value proposition text above,
plus "we can still adjust anytime right?" (incremental build). This entry
covers the whole build session, including the reported fact-sheet crash and
its fix (see below).

### What the assistant produced
Three app modules at the project root and an app-facing data layer:
- `streamlit_app.py` - the 6-page EverVest app: Home (hero with my value
  proposition verbatim, "What EverVest is", investment philosophy, fund
  catalogue, and an "Explore funds" CTA that navigates to Compare), Compare
  funds (family/type filters, net-vs-gross toggle, rankable Sharpe table,
  Sharpe chart, interactive growth-of-$1), Fund fact sheet (per-fund metrics,
  plain-English strategy blurb, current holdings with equity/crypto sleeve
  grouping for combined funds, growth + drawdown charts), Allocate (up to 5
  funds, weight sliders, blended 2023 result), News & sentiment (fusion
  before/after + lambda-sensitivity, FinVADER sector grid, VADER-vs-FinVADER),
  and About the data (methodology + limitations, not investment advice).
- `src/app_style.py` - the FT design system for the web: cream #F7F1E8
  background, #333333 text, #D9D2C7 grid, the 5-colour palette, Georgia
  headings, burgundy primary buttons, and reusable hero/stat/card HTML.
- `src/app_data.py` - cached, defensive readers over `results/` (returns,
  weights, metrics, sentiment, fusion tables) plus derived helpers: fund
  parsing into family/method/model, metrics-from-series, fund blending with
  the 252/365 annualisation rule, and current-holdings with `EQ_`/`CR_` leg
  splitting for combined funds.

### What was wrong or risky
- **Fact-sheet crash on fused funds (user-reported).** `performance_metrics.csv`
  had the 8 sentiment-fused rows with `basis` (and `n_days`) as NaN after the
  manual annotation pass, so the app found no net/gross row and crashed with
  KeyError on `ann_return`. The default fact-sheet fund was a non-fused fund,
  so the AppTest smoke run missed it - a reminder that smoke tests must
  exercise every data path, not just page loads.
- Two bugs caught by AppTest during the build: returns CSV columns are
  `net_return`/`gross_return` not `net`/`gross` (pivot KeyError); and three
  same-family funds produced identical `number_input` labels, tripping
  Streamlit's duplicate-widget-id guard (fixed with unique keys and a
  `short_name()` helper).
- Runtime pitfalls fixed along the way: AppTest does not put the script dir on
  sys.path (added explicit path insert); the fusion lambda-sensitivity table
  stores `family` lowercase (`equity`), not `Equity`; and the fusion
  before/after table uses net rows that only matched after `basis` was filled.
- The manual `performance_metrics.csv` annotations (Universe/Method/" "/
  Unnamed columns) are fragile and could be silently overwritten by a pipeline
  re-run; I did NOT re-run the pipeline.

### What I changed and why
- Data fix: filled `basis` = gross/net per pair and `n_days` = 250 for the 8
  fusion rows only, preserving every manual annotation column. The net values
  were verified against `fusion_before_after.csv` (fused_sharpe 0.6622 vs
  0.662 in the file) and recomputed independently from `fund_returns.csv`
  (ann_return 0.073, sharpe 0.662, maxDD -0.082) before writing.
- App hardening: `metric_row()` now falls back to any row for the fund, then
  to computing metrics from the precomputed returns series, so a missing/NaN
  row can never crash the fact sheet again; all metric formatters render NaN
  as "—".
- Verification: ruff clean; Streamlit AppTest passes all 6 pages and every one
  of the 19 fund fact sheets (the loop that caught this bug); app launched
  locally at http://localhost:8501 (HTTP 200) with `headless=true`. No nltk /
  finvader / data_access imports anywhere in the app (grep-verified).

---
## Entry 18 - EverVest polish: investor-friendly naming, objective-based Home, Market Insights
### What I wanted
Three follow-up requests on top of the built app, all display-layer (no fund IDs,
calculations, datasets or outputs change):
1. **Investor-friendly rename everywhere.** Equal Weight->Diversified Core,
   Minimum Variance->Capital Preservation, Maximum Sharpe->Risk-Adjusted Growth,
   Risk Parity->Balanced Risk, Minimum CVaR->Downside Protection, VADER/finVADER
   fused->News-Aware ..., Combined->Diversified (with "Equity + Crypto" as
   supporting text). Dropdown labels like "Capital Preservation - Crypto"; fund
   cards "Friendly / Technical strategy - Universe"; a plain-English description
   per strategy; consistent across navigation, Compare, fact sheets, Allocate and
   About. The research-enhanced category should show ONLY "Equity Minimum Variance
   + VADER" (not the finVADER variants).
2. **Simplify Home** so it explains strategies by objective instead of repeating
   each one per universe: replace the 15 core cards with five objective-based cards
   (Diversified Core / Capital Preservation / Risk-Adjusted Growth / Balanced Risk /
   Downside Protection), each shown once with its technical method, plain-language
   description and the three universes. Keep the "Research-enhanced strategies"
   section and the Equity News-Aware card unchanged. Page order: short value
   proposition, "Choose your investment objective", five cards, research-enhanced,
   "Compare funds" CTA. Link cards to Compare funds with the strategy (and, where
   practical, universe) preselected. Later feedback: do NOT stack the universes
   vertically as badges - list them inline like "- Equity - Crypto - Diversified"
   with a link to Compare funds.
3. **Rename "News & sentiment" -> "Market Insights"** and reorder it for long-term
   retail investors: What the news indicated / Sentiment across sectors / How news
   changes a portfolio / Did news sentiment add value? (and why VADER was used) /
   Explore the research (technical diagnostics in expanders). Add a visible
   disclaimer that sentiment is the tone of news, not live market information, and
   does not predict returns; label the 2023 evaluation period clearly; keep every
   existing sentiment analytics artifact available.

### Prompt(s)
- The naming request: map each strategy to a friendly objective name, keep fund
  IDs/columns/files unchanged, "make each universe badge or a 'View funds' button
  open Compare Funds with the relevant strategy and universe selected", and list
  the five cards with the exact descriptions and "Available in:" universes given.
- "Simplify the EverVest homepage ... Replace the existing 15 core fund cards with
  five objective-based cards ..." then "Rename the 'News & sentiment' navigation
  item and page heading to 'Market Insights'. Reorganise the page for long-term
  retail investors in this order: ..."
- Bug report: "okay i think the categorise is good, but now the card is vertical,
  i do not want that ... list out the universe, like - Equity - Crypto - Diversified
  ... however, the compare funds is not working for all of them KeyError: 'Objective'".

### What the assistant produced
- `src/app_data.py`: a display-only naming layer - `METHOD_FRIENDLY`,
  `UNIVERSE_LABEL`/`UNIVERSE_SUPPORT` (Combined->Diversified / "Equity + Crypto"),
  `STRATEGY_DESCRIPTION`, `FAMILY_ORDER`, plus `friendly_name()`, `display_lines()`,
  `dropdown_label()`, `strategy_description()`, `catalogue_group()` and `catalogue()`;
  removed `METHOD_SHORT`/`short_name()` and `funds_overview()`. `compare_table()`
  gained friendly Fund/Strategy/Universe columns and a new display-only `Objective`
  column (friendly method name) used to filter Compare.
- `src/app_style.py`: `.ever-group`, `.ever-fund` and `.ever-fund .funi` CSS plus
  `group_heading_html()` and `fund_card_html(..., footer="")` for the inline
  universe list.
- `streamlit_app.py`: Home in the required order (hero + "Choose your investment
  objective" + five cards with "Available in: - Equity - Crypto - Diversified" and a
  "View funds" button each + the unchanged research-enhanced card + CTA); Compare
  with Universe / Investment objective / Fund type filters and session-state preset
  routing; Market Insights rebuilt around the five investor questions with the
  disclaimer and diagnostics in expanders; the in-app lambda chart legend no longer
  says "chosen".

### What was wrong or risky
- **KeyError: 'Objective' on Compare funds (user-reported, live server only).** The
  running Streamlit server had been started under the system Python
  (`/Library/Frameworks/...`, not the repo `.venv`) BEFORE `src/app_data.py` gained
  the Objective column, so its in-memory `app_data` module was stale while the
  re-read `streamlit_app.py` referenced `df["Objective"]`. AppTest did not catch it
  because it launches a fresh interpreter each run - only the long-lived server
  process kept the old module. Lesson: after editing app modules, restart the
  server with the repo interpreter rather than assuming the file watcher reloads
  imported modules.
- First routing version passed the technical method ("Equal Weight") into the
  friendly objective filter ("Diversified Core"), silently producing an empty
  multiselect (caught by AppTest).
- The three stacked universe buttons per card were rejected as "vertical"; the
  replacement edit also left the old button block in the file (20 home buttons
  instead of 5) - caught by AppTest and removed.
- The CSS string lost a stray closing parenthesis (orphaned `)`) and two helpers
  (`inject_css`, `hero_html`) vanished from `src/app_style.py` during editing -
  restored and re-verified.

### What I changed and why
- Kept the rename display-only: fund IDs, dataset columns and files untouched;
  mapping documented on the About page. Verified 19 unique dropdown labels and the
  catalogue split (15 Core + 1 Research-enhanced = only "Equity Minimum Variance
  + VADER").
- Routing is a small `st.session_state` preset (objective + optional universe) that
  Compare pops before rendering its filters - no query-param or multipage refactor.
- Market Insights leads with investor conclusions; the same figures/tables (sector
  grid, VADER-vs-finVADER, lambda sensitivity, fusion before/after, weight tilt)
  remain available in the body or in expanders, so the Project B sentiment analytics
  requirement still holds. 2023 is labelled as the out-of-sample evaluation window
  and the disclaimer is a prominent `st.info` box.
- Root-caused the KeyError operationally (stale process, correct interpreter),
  not by patching around it; verified with `ps`/`lsof` and a clean restart.
- Verification: ruff clean + formatted; Streamlit AppTest passes Home, objective
  routing into Compare, Market Insights, all other pages, all 19 fund fact sheets
  and Compare default filters; server restarted at http://localhost:8501 (HTTP 200).

---

## Entry 19 - Report figure: VADER vs finVADER tilt-sensitivity (lambda sweep)
### What I wanted
A report-ready figure from the EXISTING sensitivity results - do not recalculate or
change the data. Plot net out-of-sample Sharpe against tilt strength lambda for
VADER and finVADER at lambda = 0, 0.25, 0.5, 0.75, 1.0, in the same FT style as the
other report figures, with a dashed vertical line at lambda = 0.5 labelled "Selected
tilt strength", lambda = 0 labelled "Baseline: no sentiment tilt", a legend
containing ONLY "VADER" and "finVADER" (no "lambda = 0.5 chosen"), markers at every
tested value, and a high-resolution PNG for Word. Output the underlying numbers as a
small CSV/printed table so they can be verified. Also confirm from the code why
lambda = 0.5 was selected, and do NOT call it optimal unless it was chosen
in-sample - if predetermined, say so.

### Prompt(s)
User brief: "Create a figure of the existing VADER versus finVADER tilt-sensitivity
chart for inclusion in the report. ... Plot net out-of-sample Sharpe ratio against
sentiment tilt strength ... Include lambda = 0, 0.25, 0.5, 0.75, 1.0 ... Add a
vertical dashed line at lambda=0.5, labelled 'Selected tilt strength' ... Label
lambda=0 as 'Baseline: no sentiment tilt' ... Before generating the figure, confirm
from the existing code or methodology why lambda=0.5 was selected."

### What the assistant produced
- `scripts/fig_lambda_sensitivity.py`: a standalone figure script that reads ONLY
  `results/tables/fusion_lambda_sensitivity.csv` (net rows, equity family - matching
  the app's existing sensitivity chart and the report's other fusion exhibits), draws
  both model lines with markers, reuses `src/plotting.py` for the FT house style
  (cream bg, no axis box, faint horizontal gridlines, bold left-aligned title, FT
  palette), adds the dashed lambda=0.5 line with "Selected tilt strength" and the
  "Baseline: no sentiment tilt" annotation, and exports
  `results/figures/fusion_lambda_sensitivity_equity.png` at 300 dpi. It also writes
  `results/tables/fusion_lambda_sensitivity_plotted.csv` and prints the pivot table
  of plotted values.

### What was wrong or risky
- **lambda = 0.5 is predetermined, not optimal.** `TILT_STRENGTH = 0.5`
  (`scripts/run_part_b.py`), the default in `src/fusion.py`, and the sensitivity
  docstring says "lambda was set as a fixed assumption, not fitted" - so the figure
  and report must say "predetermined", never "optimal". The lambda=0 row is an
  internal consistency check (both lexicons are identical at lambda=0 by
  construction).
- Matplotlib quirk: `ft_title()` uses `loc="left"`, so `ax.get_title()` returns the
  CENTER title (empty here) and misleads validation - the left-title artist renders
  fine and must be checked directly (or via the canvas).
- The first baseline label sat below the auto y-limits and would have been clipped;
  fixed by expanding `set_ylim` and anchoring the annotation in offset points.
- Equity family is an assumption (the combined-family rows exist in the same CSV and
  could be plotted instead if the report wants them).

### What I changed and why
- Built a standalone script (instead of editing `run_part_b.py`) so the figure is
  regenerated from the precomputed CSV with no pipeline re-run and no risk of
  touching the fused-row manual annotations.
- Reused `src/plotting.py` so the figure is visually identical in style to the other
  report exhibits.
- Plotted values (net Sharpe, equity): VADER 0.4901 / 0.5930 / 0.6622 / 0.7209 /
  0.7715 and finVADER 0.4901 / 0.4963 / 0.4532 / 0.4208 / 0.4160 at lambda 0 to 1.0;
  verified against the CSV and `fusion_before_after.csv` (0.6622 net matches).
- Verification: ruff clean + formatted; label placement validated programmatically
  (inside canvas, no line overlap, legend contains only VADER/finVADER); output is
  300 dpi, 2580x1470.

---

## Entry 20 - Make the Compare-page Sharpe chart filter-aware (live)
### What I wanted
The Compare funds page embedded the static report figure
`results/figures/sharpe_by_fund.png`, which never reacted to the page's Universe /
Objective / Fund-type filters or the net-vs-gross toggle, and was built on GROSS
returns while the table defaults to NET. The user asked to make it "alive": a chart
drawn from the same filtered, basis-correct table shown above it.

### Prompt(s)
User brief: "make it alive first" (replacing the static PNG with a chart that
responds to the filters and basis toggle).

### What the assistant produced
- Replaced the `st.image(...)` block in `page_compare()` (`streamlit_app.py`) with a
  live matplotlib grouped-bar chart: x-axis = universes present in the filtered
  table, one bar per fund in the table, grouped by universe, coloured by strategy
  (technical method, with "+ VADER" / "+ finVADER" for fused funds) using the same
  FT palette, drawn from the already-filtered `df` so it honours the Universe /
  Objective / Fund-type filters AND the net/gross basis toggle.
- Strategy ordering reuses `ad.METHODS` (method first, fused variants after), and
  each strategy keeps one colour across universes, mirroring the report exhibit's
  readability.
- Empty-filter state shows "No funds match the current filters - nothing to chart."
  instead of crashing.
- Added `import numpy as np` to `streamlit_app.py`.

### What was wrong or risky
- First attempt passed `pad=26` to `as_.ft_title()`, which takes no `pad` argument
  (`TypeError` caught by AppTest) - removed.
- A leftover scratch lambda (`_strat_key`) was dead code - removed.
- The report's static `sharpe_by_fund.png` stays untouched and still used in the
  report; the caption now says the report copy is identical, so no claims drift.

### What I changed and why
- Kept the chart logic inline in `page_compare()` (consistent with the existing
  growth-of-$1 chart there) rather than adding a new src/ helper.
- Verified: ruff check + format clean; AppTest smoke suite passes (all pages,
  Compare default filters, 19 fact sheets); numeric check confirms 19 bars default,
  filters narrow the bars, fused strategies group correctly, and net vs gross values
  differ (Equity Min Var 0.49 net vs 0.491 gross).
- Restarted the live server (kill + venv python) so the running app serves the new
  chart at http://localhost:8501 (HTTP 200).

---

## Entry 21 - Compare-page back button + smaller figure displays
### What I wanted
- Add a "go back" button on the Compare funds page so it is easy to return to
  Home and pick a different investment objective (the Home objective cards are
  the intended entry point).
- Make the standalone figure displays smaller - they were full page width
  ("too catchy").

### Prompt(s)
User brief: "add a go back button in the compare fund, because its easier to view
different objective by clicking the options in home page, and also can you make the
individual figures displays a bit smaller, its too catchy now".

### What the assistant produced
- `← Back to Home` button at the top of `page_compare()`, rendered inside a narrow
  left column so it is compact, wired through the existing `go("Home")` router.
- New `centred_media(ratio=0.62)` helper: a centred Streamlit column of ~62% page
  width used to wrap every standalone figure (pyplot and PNG), so charts no longer
  span the full wide layout.
- Shrank the matplotlib figures that looked oversized: Compare Sharpe chart
  (9.8x5.4 -> 7.6x4.2), Compare growth chart (9.2x4.4 -> 7.6x3.8), Market Insights
  lambda chart (9.2x4.2 -> 7.6x3.6).
- Applied the centred smaller display to all standalone figures: Compare Sharpe +
  Growth charts, fact-sheet combined weights PNG, Market Insights sector grid,
  weight-tilt, fused-Sharpe, VADER-vs-finVADER and sector-index-over-time PNGs, and
  the lambda chart. The fact-sheet 2-up (growth/drawdown) and Allocate blended
  chart stay in their existing column layouts.

### What was wrong or risky
- Growth-of-$1 title hardcoded "(net of costs)" while the chart already follows the
  net/gross toggle - made it basis-aware (`basis_lbl`).
- First lint pass flagged `E501` on the fused-Sharpe caption line and `SIM117`
  (nested `with` in the two-column expander); fixed by wrapping the caption and
  combining `with c1, centred_media(...)`.

### What I changed and why
- Kept the helper local to `streamlit_app.py` (used only by app pages) rather than
  moving it into `src/`.
- Used `use_container_width`-style container scaling (narrower column) plus smaller
  figsize so both in-app charts and precomputed PNGs shrink consistently without
  touching any `results/figures` asset.
- Verified: ruff clean + formatted; AppTest smoke suite passes (all pages,
  objective routing to Compare, Compare defaults, 19 fact sheets); server restarted
  and live at http://localhost:8501 (HTTP 200).

---

## Entry 22 - Arrow-only back control, slightly narrower standalone charts, research-card link
### What I wanted
Three targeted UI changes (no redesign of unrelated parts):
1. Replace the text "Back to Home" button with a single `←` icon at the top-left
   of the main content, before the page heading, without pushing it, with an
   accessible tooltip "Back to Home", consistent placement on all relevant pages.
2. Slightly reduce ONLY standalone chart widths to ~85-90% of available width,
   centred; keep the sector sentiment score figure and side-by-side figures at
   their current sizes; no global CSS rule; no changes to tables/cards/data.
3. Link the "Equity News-Aware Capital Preservation" card (technical fund
   "Equity Minimum Variance + VADER") so a "View fund" action navigates to the
   Fund fact sheet with that fund preselected, reusing the existing
   session-state routing used by the objective cards.

### Prompt(s)
User brief (verbatim core): "Replace 'Back to Home' with an arrow-only control...
single icon `←`... top-left of the main page content... before the page heading
without pushing... accessible tooltip or label 'Back to Home'... same placement
and appearance consistently across all relevant pages." + "Slightly reduce the
width of standalone charts... approximately 85-90%... Do not resize the sector
sentiment score figure... Do not resize figures displayed side-by-side... Do not
apply a global CSS rule." + "Link the Research-enhanced strategy to its assigned
fund... Navigate to the Fund Fact Sheet page. Automatically select 'Equity
Minimum Variance + VADER'... Reuse the existing navigation and session-state
approach... Do not rename the underlying fund ID or modify its data."

### What the assistant produced
- `page_header(title)` helper: heading rendered in a `st.columns([0.06, 0.94])`
  row with an arrow-only `st.button("←", help="Back to Home")` on the left, so
  the arrow sits before (to the left of) the heading with zero vertical push.
  Applied to all non-Home pages: Compare funds, Fund fact sheet, Allocate,
  Market Insights, About the data. The old Compare-only "← Back to Home" text
  button was removed.
- `open_fact_sheet(fund)` mirroring `open_compare()`: sets
  `st.session_state.fact_sheet_fund` and routes via `go("Fund fact sheet")`.
  `page_fact_sheet` pops the preset into a keyed selectbox
  (`st.selectbox(..., key="sheet_fund")`), so the fund is auto-selected and its
  fact sheet renders immediately.
- Home research section: each research card now has a "View fund" button wired to
  `open_fact_sheet(f["id"])` (fund ID untouched: "Equity Minimum Variance +
  VADER"), same navigation behaviour as the objective cards' "View funds".
- `centred_media` default ratio 0.62 -> 0.88 (standalone charts now ~88% of
  width, centred, small but readable reduction). The sector sentiment grid is
  pinned to `ratio=0.62` to keep its current size; the two side-by-side
  sector-index images keep `ratio=0.92`. No tables/cards/non-chart elements
  touched, no global CSS resize.

### What was wrong or risky
- The old smoke test asserted exactly 5 `home_` buttons; the research card adds a
  sixth ("View fund"). Test updated to 6 and extended with a routing check
  (research button -> Fund fact sheet with "Equity Minimum Variance + VADER"
  selected), which passes.
- `page_header` uses `vertical_alignment="center"` (Streamlit 1.58 supports it)
  so the arrow and heading align without a vertical gap.

### What I changed and why
- Reused the existing `go()`/session-state pattern everywhere rather than a
  custom CSS/JS clickable card, so navigation stays consistent and testable.
- Narrowed charts by container ratio only (no data/style changes), satisfying
  "no global CSS rule" - each standalone figure is wrapped individually.
- Verified: ruff clean + formatted; AppTest smoke suite passes (Home buttons,
  objective routing, research-card routing, Market Insights, all pages, 19 fact
  sheets, Compare defaults); per-page check confirms exactly one `←` arrow with
  help "Back to Home" on every non-Home page and that clicking it returns to
  Home; server restarted, live at http://localhost:8501 (HTTP 200).

---

## Entry 23 - Full-size sector sentiment figure, Methodology tab, data-scope review
### What I wanted
1. Restore the "Sentiment across sectors" figure (`sentiment_sector_grid_finvader.png`)
   to its ORIGINAL full width - the previous change had pinned it inside
   `centred_media(ratio=0.62)`, making it smaller than the original display.
2. Add one new app tab called "methodology" collecting every equation used in the
   analysis in the app.
3. Review `src/data_access.py`, `context/DATA_GUIDE.md`, and the project context /
   brief for anything missed or out of data scope, and report the findings.

### Prompt(s)
User brief (verbatim core): "the figures of sentiment across sectors should be the
original size, do not make it smaller" + "add one tab called methodology. Collect
all the equation used for the analysis in the app. and collect them in the new
tab" + "review data_access.py, the data guide and project context to see if we
missed anything, and to ensure nothing is out of data scope. tell me what you
find, and if unsure note it as uncertain."

### What the assistant produced
- Sector grid: removed the `centred_media(ratio=0.62)` wrapper around
  `sentiment_sector_grid_finvader.png`; it is again a full-width
  `st.image(..., width="stretch")` (the side-by-side sector-index pair stays at
  ratio=0.92; standalone charts stay at 0.88).
- New `page_methodology()` registered as "Methodology" in `PAGES` (between Market
  Insights and About the data) and in `PAGE_FNS`, using the shared `page_header`.
  It documents the implemented pipeline in 8 numbered sections plus a key
  assumptions list, with 17 LaTeX equations: (1) simple daily returns within
  panel + 252/365 annualisation + calendar merge/cap; (2) walk-forward expanding
  window with no look-ahead, first live backtest date 2023-01-03 (250 days) /
  2023-01-01 (365 days); (3) the five optimisation methods (equal weight, minimum
  variance, maximum Sharpe with rf=0, risk parity via log-barrier, minimum CVaR
  alpha=0.95 LP); (4) growth-of-$1, annualised return/vol, Sharpe, max drawdown;
  (5) 5bps one-way turnover costs and net return; (6) VADER/finVADER compounds,
  ticker-day means, equal-weight + coverage-weighted sector index, 1-day lag,
  21-day rolling grid; (7) cross-sectional z-score fusion tilt scaled by
  lambda=0.5 (predetermined); (8) capital-weighted blend with 365-day rule.
- Fixed an out-of-scope claim found in review: the About page said equity has
  "7 sectors" but the data guide states 50 US large-caps across **10 sectors**;
  corrected to "10 sectors".

### What was wrong or risky
- AppTest in this Streamlit version does not expose image width (the `imgs` proto
  only carries url/caption), so width could not be asserted programmatically;
  verified by source inspection instead (line 577-581, `width="stretch"`).
- First subheader-count assertion expected 8 sections but there are 9 ("Key
  assumptions" is a 9th `st.subheader`); corrected the test.

### What I changed and why
- Restored the grid because the user asked for the original size; the report
  exhibit (300dpi `sharpe_by_fund.png` etc.) is untouched.
- Methodology reflects only equations actually implemented in `src/` (etl,
  portfolios, fusion, sentiment, app_data) - nothing invented; lambda=0.5 stated
  as a predetermined assumption, not optimal.
- Scope review summary (reported to user): data_access.py is the frozen provided
  loader (3 datasets: 50-equity/10-sector panel, 10 cryptos, news 2020-2023) and
  was not edited; the app reads only `results/` artifacts; crypto 2024-01-01
  stray rows are capped at 2023-12-31 per the data guide; nothing else in the
  app goes outside the supplied data scope.
- Verified: ruff clean + formatted; full AppTest smoke suite passes (6 Home
  buttons, objective + research routing, all pages incl. Methodology, 19 fact
  sheets, Compare defaults); Methodology page renders 17 latex blocks / 9
  subheaders with no exceptions; server restarted, live at http://localhost:8501
  (HTTP 200).

---

## Entry 24 - NumberColumn format bug: literal ".2f" rendered in table cells
### What I wanted
The Compare-page Sharpe column and the Market Insights sentiment before/after
table showed the literal text ".2f" in every cell instead of the value (e.g.
"0.49").

### What the assistant produced (root cause)
Streamlit 1.58's `st.column_config.NumberColumn(format=...)` uses
**printf-style (sprintf.js) format specifiers** - e.g. `"%.2f"` - plus named
keywords ("percent", "plain", ...). My code used `".2f"` (the old d3-format
style). `.2f` is not a printf conversion specifier, so sprintf.js passes it
through as literal text. Verified against the installed package docstring
(`elements/lib/column_types.py`, format section lists printf-style like
`"£ %.2f"`) and by decoding the serialised column config JSON in the AppTest
proto, which carried `"format": ".2f"`.

### What I changed and why
- `streamlit_app.py`: `format=".2f"` -> `format="%.2f"` for the Compare Sharpe
  column and for `baseline_sharpe` / `fused_sharpe` / `delta_sharpe` in the
  "Did news sentiment add value?" table. The `percent` keyword columns were
  already correct.
- Verified the proto now carries `"%.2f"` for all four columns; ruff clean;
  smoke suite passes; server restarted (HTTP 200).

### What was wrong or risky
- Earlier "fixes" had only toggled `.2f` vs `.3f`, which never addressed the
  real bug (both are invalid printf specifiers and would still render
  literally). The correct format is `%.2f` for 2 decimals / `%.3f` for 3.
- AppTest does not render cells (no HTML), so the literal text had to be
  explained from the sprintf.js spec + the docstring; the serialised config
  JSON was used as the wire-level check.

---

## Entry 25 - Key findings + sector summary table polish in the remaining Market Insights sections
### What I wanted
Improve the three remaining research sections in Market Insights: add one
short, plain-language "Key finding" above each figure/table, rename the sector
summary table headings, and show the positive-day columns as percentages.
Display only - no changes to calculations, figures, or underlying data.

### What I investigated first
I can't view the figures (no image input in this model), so I analysed the
underlying CSVs the figures are built from:
- `results/data/sector_sentiment_index.csv` (the two over-time line charts):
  overall mean sentiment positive for both models (VADER 0.085, finVADER
  0.054); sector means 0.03-0.10; Materials lowest under BOTH models (VADER
  0.052, finVADER 0.032).
- `results/tables/fusion_lambda_sensitivity.csv` (the tilt-strength chart):
  VADER net Sharpe rises monotonically 0.49 (lambda=0) to 0.77 (lambda=1);
  finVADER falls to 0.42 (lambda=1). The plotted CSV confirms lambda=0.5
  (the predetermined app setting) sits between these.
- `results/tables/sector_sentiment_summary.csv` (the table): positive days
  range 65.4%-90.7% (VADER) and 69.0%-87.1% (finVADER) across all 10 sectors;
  Materials lowest average under both models.

### What I changed
- "Sentiment index over time, by sector (2020-2023)": key-finding markdown
  above the two charts (sectors moved together, stayed mostly positive,
  Materials least positive under both models).
- "How sensitive is the tilt to its strength?": key-finding markdown above the
  chart (VADER 0.49->0.77, finVADER ->0.42, lambda=0.5 fixed in advance).
- "Sector sentiment summary (table)": key-finding markdown above the table
  (positive on the large majority of days everywhere; Materials most negative
  on average under both models).
- Sector table headings renamed:
  sector -> Sector, n_days -> Days observed, vader_mean -> VADER average,
  vader_std -> VADER variation, vader_pct_positive -> VADER positive days,
  finvader_mean -> finVADER average, finvader_std -> finVADER variation,
  finvader_pct_positive -> finVADER positive days.
- Positive-day columns formatted with `cc.NumberColumn(format="percent")`.

### What was wrong or risky
- Streamlit's `percent` column-config format uses Intl.NumberFormat percent
  style (verified in the bundled frontend), which multiplies by 100, so a
  0.8856 stored fraction renders as "88.56%". That matches the stored decimals
  without touching the data. The proto check confirmed `"format": "percent"`
  reaches the frontend.
- Renaming via `.rename(columns=...)` keeps the cacheable loader untouched;
  no data mutation.

### What I checked
- AppTest (Market Insights): no exceptions; all four key findings present
  (the existing VADER vs finVADER one plus three new ones); sector table shows
  the new headings and the percent column config in the serialised proto.
- Key-finding numbers verified against the raw CSVs (Materials lowest mean in
  both models; VADER/finVADER Sharpe endpoints at lambda 0/1).
- ruff format + ruff check clean; full smoke suite passes; server restarted
  (HTTP 200).

--- 

## Entry 26 - Verify sector_sentiment_index() against the report claims: coverage weighting + lag safety
### What I wanted
Before writing the Section 3/4 narrative I asked my assistant to verify the
actual `sector_sentiment_index()` code in `src/sentiment.py` against what the
report intends to claim, because I believed the report/CLAUDE.md described it
as HEADLINE-COVERAGE-WEIGHTED. Two things to check, in order:
1. Does the sector aggregation weight each ticker by its daily headline count
   (`weighted_mean = sum(score*n_headlines)/sum(n_headlines)`), or is it a
   plain equal-weight mean with no headline-count term?
2. Is the `groupby("sector").shift(lag)` positional lag safe, or does it skip
   trading days for the thin sectors (Materials, Utilities, Real Estate) whose
   news coverage is documented as sparse?

And, before any code change: a before/after comparison (equal-weight vs
coverage-weighted) for those three sectors, so I could judge whether the
numbers move enough to rewrite the Section 4 narrative.

### Prompt(s)
"Read src/sentiment.py in full, specifically sector_sentiment_index() around
line 135, before making any changes. Check (1) whether ticker_day carries a
headline-count column at all and whether the sector aggregation is a
headline-count-weighted mean; if not, trace back to where per-ticker-day scores
are first computed and add n_headlines there. Check (2) whether any (date,
sector) combination in the expected trading calendar is missing BEFORE the lag
is applied - reindex against the full grid, count the missing combos broken
down by sector - because a gap would make shift(1) quietly become lag 2+.
Show me both functions side by side; trace which one is actually called through
run_part_b.py / fusion / results/data/*.csv; tell me which function produced
the numbers currently in the report and whether the other is dead code. Get the
before/after comparison for Materials, Utilities and Real Estate FIRST, and
report back before touching any code."

### What the assistant produced (my corrections to its findings)
The assistant showed me both functions side by side and confirmed the
coverage-weighted function `coverage_weighted_sector_sentiment_index`
(`src/sentiment.py:196`) genuinely implements `sum(score*n)/sum(n)` using
`n_headlines` from `ticker_day_coverage` - it is not a cosmetic reweighting.
Then it traced the wiring and found my original premise was wrong:
- `sector_sentiment_index` (equal-weight, `src/sentiment.py:135`) is the
  PRIMARY index: it writes the required `results/data/sector_sentiment_index.csv`
  that the app reads (`src/app_data.py:101`) and check_handin.py requires.
- The coverage-weighted function IS called by both pipelines but its artifact
  `coverage_weighted_sentiment_index.csv` has ZERO readers anywhere in the
  repo - write-only, disconnected from the app, fusion, and all figures.
- Fusion (`src/fusion.py:43`) consumes RAW ticker-day sentiment
  (`run_part_b.py:718` passes `ticker_day` into `build_funds`), so neither
  sector index feeds fusion; Section 4 numbers are unaffected by this choice.
- Nothing in CLAUDE.md or the (still OUTLINE-only) report describes
  `sector_sentiment_index` as coverage-weighted; the module docstring and
  prompt log document equal-weight as primary and coverage as a separate option.

I asked it to run the diagnostic anyway. Reproducible script
`scripts/sentiment_coverage_comparison.py` + two new tables:
`results/tables/sentiment_equal_vs_coverage.csv` and
`results/tables/sector_headline_coverage.csv`.

### What was wrong or risky (what I had to fix in my own assumptions)
- My assumption that the report/CLAUDE.md describe coverage weighting was
  wrong; the real finding is the opposite: the coverage-weighted implementation
  is correct but orphaned (its output is never read). This is an incomplete
  wiring, not a missing feature and not a bug in the math.
- The positional-lag concern did NOT materialise with the current wiring: the
  pre-lag sector grid is COMPLETE (10,060/10,060, 0 missing date-sector
  combos), because no-news ticker-days are filled neutral, so
  `groupby(sector).shift(1)` is a true one-trading-day lag for both indices.
  It would only become unsafe if `ticker_day_sentiment(..., fill_missing="drop")`
  were used - worth a caveat line in the report.
- The numbers DO move a lot for the three thin sectors, so the equal-weight
  choice is not robust in MAGNITUDE:
  - Coverage: Materials 5,393 headlines (3.7% of all, news on 93.4% of days),
    Real Estate 5,339 (3.6%, 93.4%), Utilities 6,319 (4.3%, 95.6%) vs Tech
    26,638 (18.1%). 228/10,060 sector-days have zero headlines.
  - Utilities VADER mean 0.109 -> 0.191 under coverage weighting, std 0.094 ->
    0.153; Materials VADER 0.055 -> 0.098, std 0.071 -> 0.138; Real Estate
    VADER 0.080 -> 0.135, std 0.093 -> 0.150.
  - Correlations equal vs coverage: 0.73-0.81 for these sectors (vs 0.85-0.88
    for well-covered Financials); mean absolute daily difference 0.052-0.093.
  - BUT sign flips are rare (1.8-3.9% of shared days, below the 6.7-9.3% seen
    in well-covered sectors): weighting changes magnitude and noise, rarely
    direction.

### What I changed and why
- Kept `sector_sentiment_index` as the equal-weight primary index and did NOT
  rewire fusion. The coverage-weighted index adds a robustness caveat, not a
  better primary signal, and rewiring would re-run the backtest and change
  numbers already in the report for a signal granularity (sector) that does not
  match per-ticker weight tilts.
- Added the reproducible comparison script + two tables above; `ruff check`
  clean; re-ran twice with identical output.

### Open questions for the report (primary sentiment index) - need my decision
1. PRIMARY: confirm the report will present the equal-weight index as the
   required `sector_sentiment_index.csv` and describe coverage weighting only
   as a robustness check, not as the headline index.
2. COVERAGE VALUE: do we keep `coverage_weighted_sentiment_index.csv` as a
   write-only artifact at all, or drop it / wire it into the app as a
   robustness panel so it is actually used? (Currently nothing consumes it.)
3. LAG CAVEAT: state explicitly in the report that the index is defined on the
   full equity calendar (neutral fill) so the one-day lag is exact, and that
   this guarantee would break under `fill_missing="drop"`.
4. ROBUSTNESS STATEMENT: use the comparison numbers above to say thin-sector
   magnitudes/volatility are weighting-sensitive while sign/direction is not -
   do I want this in Section 3 (sentiment) or Section 6 (critical reflection)?

