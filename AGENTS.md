
# Project Context
- Assisting with FINS3645 Project B (Stations 3-4: funds, backtest, sentiment, app). Part A (Stations 1-2) is done and its tested modules are reused here.
- Follow PROJECT_BRIEF.md and context/ before recommending anything.
- Don't assume requirements not explicit or reasonably implied by the brief.
- If my idea conflicts with the brief, data, or rubric, explain the conflict and propose an alternative rather than just agreeing.
- Load data only via src/data_access.py. Never download, copy, or commit raw data files. Never assume additional data exists.

# Backtest & Fund Mechanics (brief-specific, easy to get wrong)
- No look-ahead: weights at time t must be formed ONLY from data available before t. The out-of-sample period starts AFTER an initial estimation window, not on the first date in the data - state the first live backtest date and window length explicitly.
- Rebalance monthly or less often (not daily) - state the exact convention (e.g. first trading day of each month).
- Annualise equity with sqrt(252), crypto with sqrt(365) - do not use one factor for a combined fund; annualise each leg with its own factor before combining, or state clearly how a blended factor was derived.
- Risk-free rate: state the choice explicitly (zero is acceptable) before computing Sharpe ratios.
- Treat each (asset family, optimisation method) pair as ONE fund, e.g. "Combined Minimum-Variance" - that's what a user invests in and what one fact sheet covers.
- Sanity-check optimiser output: solvers on tiny daily-return covariances can silently stall. Confirm weights actually differ across methods before trusting any result.

# Assistant Rules
- Prioritise assessment requirements before optional improvements.
- If a request is outside Part B scope, explain why before proceeding.
- Explain why code works, not just generate it.
- Never delete or modify data without justification and my agreement.
- Be critical rather than agreeing with unsupported ideas.

# Coding Conventions
- Reusable functions in src/, executable scripts in scripts/.
- Figures to results/figures/, tables to results/tables/, app-readable data artifacts to results/data/.
- Modular, reproducible code. Never modify raw datasets.

# Quality Control
- Never invent citations, statistics, or results.
- Never assume generated outputs are correct - verify important ones, including by re-running the full pipeline after any change, not just checking syntax.
- Highlight assumptions and limitations. Point out errors or risky reasoning in my ideas.

# Verification and Correction Process
- Compare generated code and recommendations against `PROJECT_BRIEF.md`, `context/`, and the marking rubric.
- Run and test all generated code before accepting it.
- Manually verify selected calculations, dates, portfolio weights, return alignment, and lag logic.
- Compare outputs across optimisation methods to confirm they are genuinely different.
- Record substantial prompts, AI outputs, identified issues, corrections, and reasons in `ai/`.
- If no error is found, document what was checked rather than inventing a correction.

# App and Deployment Rules
- The Streamlit app must load precomputed artifacts from `results/`; it must not rerun backtests or VADER.
- Preserve the required filenames:
  - `results/data/fund_returns.csv`
  - `results/data/fund_weights.csv`
  - `results/data/sector_sentiment_index.csv`
  - `results/tables/performance_metrics.csv`
- Keep the app lightweight and ensure it supports fund comparison, fact sheets, allocation, and sentiment analytics.
