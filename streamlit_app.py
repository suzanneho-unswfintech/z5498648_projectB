"""EverVest - long-term, evidence-based investing app.

A thin Streamlit reader over the precomputed results/ artifacts for Part B.
It never recomputes backtests or sentiment: all data loads are cached from
results/, and charts either reuse the FT-style PNGs or are drawn in-app with
the same design tokens. No sentiment-scoring or data_access imports.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import src.app_data as ad
import src.app_style as as_
import streamlit as st
from streamlit import column_config as cc

VALUE_PROPOSITION = (
    "EverVest's value proposition is therefore to help long-term retail "
    "investors compare diversified, systematic investment opportunities by "
    "integrating historical returns, downside risk and company news into a "
    "single evidence-based decision-support platform."
)

PAGES = [
    "Home",
    "Compare funds",
    "Fund fact sheet",
    "Allocate",
    "Market Insights",
    "Methodology",
    "About the data",
]

LIMITS = (
    "All strategies are built on a 2020-2022 in-sample window and evaluated "
    "out-of-sample over 2023 (252 trading days for equity/combined funds, 365 "
    "for crypto). Returns are net of a 5bps one-way cost on rebalance trades "
    "with the risk-free rate set to zero. This is coursework, not investment "
    "advice: a single out-of-sample year cannot prove a strategy works."
)

# Investment objectives: each one maps to a technical method available in all
# three universes (display-layer only - fund IDs and data are unchanged).
OBJECTIVES = [
    {
        "name": "Diversified Core",
        "method": "Equal Weight",
        "desc": (
            "Spreads money equally across all available assets, providing a "
            "simple and transparent starting point."
        ),
        "universes": ["Equity", "Crypto", "Diversified"],
    },
    {
        "name": "Capital Preservation",
        "method": "Minimum Variance",
        "desc": (
            "Prioritises lower volatility and a steadier investment journey "
            "rather than chasing the highest return."
        ),
        "universes": ["Equity", "Crypto", "Diversified"],
    },
    {
        "name": "Risk-Adjusted Growth",
        "method": "Maximum Sharpe",
        "desc": ("Seeks stronger returns while considering the amount of investment risk taken."),
        "universes": ["Equity", "Crypto", "Diversified"],
    },
    {
        "name": "Balanced Risk",
        "method": "Risk Parity",
        "desc": (
            "Distributes risk more evenly so that highly volatile assets do "
            "not dominate the portfolio."
        ),
        "universes": ["Equity", "Crypto", "Diversified"],
    },
    {
        "name": "Downside Protection",
        "method": "Minimum CVaR",
        "desc": ("Focuses on limiting losses during the portfolio's worst market periods."),
        "universes": ["Equity", "Crypto", "Diversified"],
    },
]
OBJECTIVE_NAMES = [o["name"] for o in OBJECTIVES]


def fmt_pct(x: float, sign: bool = False) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:+.1%}" if sign else f"{x:.1%}"


def fmt_num(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


def go(page: str) -> None:
    st.session_state.nav = page
    st.session_state.page = page
    st.rerun()


def open_compare(objective: str, universe: str | None = None) -> None:
    """Open Compare funds with one objective (and optionally one universe) preselected."""
    st.session_state.compare_preset = {"strategy": objective, "universe": universe}
    go("Compare funds")


def open_fact_sheet(fund: str) -> None:
    """Open the Fund fact sheet with one fund preselected."""
    st.session_state.fact_sheet_fund = fund
    go("Fund fact sheet")


def sidebar_nav() -> str:
    with st.sidebar:
        st.title("EverVest")
        st.caption("Long-term, evidence-based investing")
        selected = st.radio("Navigate", PAGES, key="nav")
        st.session_state.page = selected
        st.markdown("---")
        st.caption(
            "Part B - systematic funds, out-of-sample 2023. Built from precomputed results only."
        )
    return selected


def basis_control(key: str) -> str:
    choice = st.segmented_control(
        "Return basis", ["Net of costs", "Gross"], default="Net of costs", key=key
    )
    return "net" if choice == "Net of costs" else "gross"


def centred_media(ratio: float = 0.88):
    """A centred container of ~`ratio` page width for figures (smaller display)."""
    side = round((1.0 - ratio) / 2, 3)
    return st.columns([side, ratio, side])[1]


def page_header(title: str) -> None:
    """Page heading with a top-left arrow-only control back to Home.

    Same placement and appearance on every page: the arrow sits to the left
    of the heading (no vertical push) and carries an accessible tooltip.
    """
    arrow_col, title_col = st.columns([0.06, 0.94], vertical_alignment="center")
    with arrow_col:
        st.button("←", key=f"back_{title}", on_click=go, args=("Home",), help="Back to Home")
    with title_col:
        st.title(title)


def stat_row(stats: dict) -> None:
    cols = st.columns(4)
    labels = [
        ("Ann. return", fmt_pct(stats["ann_return"], sign=True)),
        ("Ann. volatility", fmt_pct(stats["ann_vol"])),
        ("Sharpe (rf = 0)", fmt_num(stats["sharpe"], 2)),
        ("Max drawdown", fmt_pct(stats["max_drawdown"])),
    ]
    for col, (label, value) in zip(cols, labels):
        with col:
            st.markdown(as_.stat_html(label, value), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------


def page_home() -> None:
    st.markdown(
        as_.hero_html(
            "EverVest", "Diversified, systematic investing made transparent", VALUE_PROPOSITION
        ),
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='ever-rule'>", unsafe_allow_html=True)

    st.subheader("Choose your investment objective")
    st.markdown(
        "Every objective is available across the **equity**, **crypto** and "
        "**diversified** (equity + crypto) universes. Pick an objective, then "
        "open a universe to see how it performed."
    )
    for obj in OBJECTIVES:
        card_col, action_col = st.columns([0.8, 0.2], vertical_alignment="center")
        with card_col:
            st.markdown(
                as_.objective_card_html(obj["name"], obj["method"], obj["desc"], obj["universes"]),
                unsafe_allow_html=True,
            )
        with action_col:
            st.button(
                "View funds",
                key=f"home_{obj['method']}_view",
                on_click=open_compare,
                args=(obj["name"],),
            )

    st.markdown(as_.group_heading_html("Research-enhanced strategies"), unsafe_allow_html=True)
    for f in ad.catalogue()["Research-enhanced strategies"]:
        st.markdown(
            as_.fund_card_html(f["name"], f["subtitle"], f["desc"]),
            unsafe_allow_html=True,
        )
        st.button(
            "View fund",
            key=f"home_research_{f['id'].replace(' ', '_')}_view",
            on_click=open_fact_sheet,
            args=(f["id"],),
        )
    st.caption(
        "The remaining sentiment variants are available in Compare funds, "
        "Fund fact sheet and Allocate."
    )

    st.markdown("")
    st.markdown("Ready to see how the strategies compare?")
    st.button("Compare funds", type="primary", on_click=go, args=("Compare funds",))


def page_compare() -> None:
    page_header("Compare funds")
    st.markdown("Filter the field, then compare the top strategies on risk-adjusted return.")

    preset = st.session_state.pop("compare_preset", None)
    if preset is not None:
        st.session_state["compare_obj"] = [preset["strategy"]]
        if preset.get("universe"):
            st.session_state["compare_unis"] = [preset["universe"]]
            st.success(
                f"Showing **{preset['strategy']}** across the "
                f"**{preset['universe']}** universe - adjust the filters anytime."
            )
        else:
            st.session_state["compare_unis"] = list(ad.FAMILY_ORDER)
            st.success(f"Showing **{preset['strategy']}** - adjust the filters anytime.")
    else:
        st.session_state.setdefault("compare_unis", ad.FAMILY_ORDER)
        st.session_state.setdefault("compare_obj", OBJECTIVE_NAMES)

    universes = st.multiselect("Universes", ad.FAMILY_ORDER, key="compare_unis")
    objectives = st.multiselect("Investment objective", OBJECTIVE_NAMES, key="compare_obj")
    kinds = st.multiselect(
        "Fund type",
        ["Core", "CVaR extension", "Sentiment-fused"],
        default=["Core", "CVaR extension", "Sentiment-fused"],
    )
    basis = basis_control("compare_basis")

    df = ad.compare_table(basis=basis)
    df = df[
        df["Universe"].isin(universes) & df["Objective"].isin(objectives) & df["Type"].isin(kinds)
    ].copy()
    df = df.sort_values("Sharpe", ascending=False).reset_index(drop=True)

    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_order=[
            "Fund",
            "Strategy",
            "Objective",
            "Universe",
            "Type",
            "Ann. return",
            "Ann. volatility",
            "Sharpe",
            "Max drawdown",
            "Days",
        ],
        column_config={
            "Ann. return": cc.NumberColumn(format="percent"),
            "Ann. volatility": cc.NumberColumn(format="percent"),
            "Sharpe": cc.NumberColumn(format="%.2f"),
            "Max drawdown": cc.NumberColumn(format="percent"),
        },
    )
    st.caption(
        "Annualised over the 2023 out-of-sample year, rf = 0, net of "
        "5bps rebalancing costs. Sort order: Sharpe, then maximum "
        "drawdown. 'Diversified' funds hold both legs: equity + crypto. "
        "Note: Combined Minimum Variance and Combined Minimum CVaR hold "
        "~0% crypto in this sample, so their figures equal their Equity "
        "counterparts."
    )

    st.subheader("Sharpe ratio by universe and strategy")
    if df.empty:
        st.caption("No funds match the current filters - nothing to chart.")
    else:
        _fig, _ax = as_.ft_figure((7.6, 4.2))
        _uni_order = [u for u in ad.FAMILY_ORDER if u in set(df["Universe"])]
        chart = df.copy()
        chart["_strategy"] = chart["_id"].map(
            lambda f: (lambda p: p["method"] + (f" + {p['model']}" if p["model"] else ""))(
                ad.parse_fund(f)
            )
        )
        _strats: list[str] = []
        for m in ad.METHODS:
            for s in chart["_strategy"].unique():
                if s not in _strats and (s == m or s.startswith(m + " + ")):
                    _strats.append(s)
        for s in chart["_strategy"].unique():
            if s not in _strats:
                _strats.append(s)
        n = len(_strats)
        _width = 0.9 / n
        _off = {s: (i - (n - 1) / 2) * _width for i, s in enumerate(_strats)}
        _x = np.arange(len(_uni_order))
        for _i, _u in enumerate(_uni_order):
            _sub = chart[chart["Universe"] == _u]
            for _s in _strats:
                _row = _sub[_sub["_strategy"] == _s]
                if _row.empty:
                    continue
                _ax.bar(
                    _x[_i] + _off[_s],
                    float(_row["Sharpe"].iloc[0]),
                    width=_width,
                    color=as_.FT_PALETTE[_strats.index(_s) % len(as_.FT_PALETTE)],
                    label=_s if _i == 0 else None,
                )
        _ax.axhline(0, color=as_.FT_TEXT, lw=0.8)
        _ax.set_xticks(_x)
        _ax.set_xticklabels(_uni_order, fontsize=10)
        _ax.set_xlabel("Asset universe", color=as_.FT_TEXT, fontsize=9)
        _ax.set_ylabel("Out-of-sample Sharpe Ratio (rf = 0)", color=as_.FT_TEXT, fontsize=9)
        _basis_lbl = "Net of costs" if basis == "net" else "Gross of costs"
        as_.ft_title(_ax, f"Sharpe ratio by universe and strategy, 2023 ({_basis_lbl})")
        if n > 1:
            _leg = _ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=min(n, 5),
                frameon=False,
                fontsize=8,
            )
            for _t in _leg.get_texts():
                _t.set_color(as_.FT_TEXT)
        with centred_media():
            st.pyplot(_fig)
    st.caption(
        "This chart responds to the filters and net/gross basis above - one "
        "bar per fund in the table, grouped by universe. The identical static "
        "exhibit also appears in the report."
    )

    st.subheader("Growth of $1 in the year")
    top_ids = df.head(5)["_id"].tolist()
    ids = df["_id"].tolist()
    selected = st.multiselect("Funds to plot", ids, default=top_ids, format_func=ad.dropdown_label)
    if selected:
        fig, ax = as_.ft_figure((7.6, 3.8))
        for i, fund in enumerate(selected):
            g = ad.growth_series(fund, basis=basis)
            ax.plot(
                g.index,
                g.values,
                lw=1.6,
                color=as_.FT_PALETTE[i % len(as_.FT_PALETTE)],
                label=f"{ad.dropdown_label(fund)} → {g.iloc[-1]:.2f}x",
            )
        ax.set_ylabel("Growth of $1")
        basis_lbl = "net of costs" if basis == "net" else "gross of costs"
        as_.ft_title(ax, f"Cumulative growth, 2023 ({basis_lbl})")
        ax.legend(fontsize=8, frameon=False)
        with centred_media():
            st.pyplot(fig)
    st.markdown(
        f"<div class='ever-muted' style='font-size:0.85rem'>{LIMITS}</div>", unsafe_allow_html=True
    )


def page_fact_sheet() -> None:
    page_header("Fund fact sheet")
    funds = ad.all_funds()
    preset = st.session_state.pop("fact_sheet_fund", None)
    if preset is not None:
        st.session_state["sheet_fund"] = preset
    fund = st.selectbox("Choose a fund", funds, format_func=ad.dropdown_label, key="sheet_fund")
    basis = basis_control("sheet_basis")
    p = ad.parse_fund(fund)
    kind = ad.fund_kind_label(fund)

    stats = ad.metric_row(fund, basis=basis)
    stat_row(stats)

    st.markdown("<hr class='ever-rule'>", unsafe_allow_html=True)
    c_text, c_hold = st.columns([1.1, 1])
    with c_text:
        head, sub = ad.display_lines(fund)
        st.markdown(as_.fund_card_html(head, sub, ""), unsafe_allow_html=True)
        st.markdown(ad.strategy_description(fund))
        st.markdown(f"**Type:** {kind}.")
        if p["model"]:
            st.markdown(
                f"Sentiment model: **{p['model']}** - monthly weights are "
                "tilted by one-day-lagged news scores."
            )
        if p["family"] == "Combined":
            st.markdown(
                "A diversified fund: it holds both legs in one portfolio, "
                "with equity and crypto tickers grouped into two sleeves "
                "(`EQ_`/`CR_`), rebalanced monthly."
            )

    with c_hold:
        st.subheader("Current holdings")
        holdings = ad.current_holdings(fund, top_n=10)
        st.dataframe(
            holdings,
            hide_index=True,
            width="stretch",
            column_config={"Weight": cc.NumberColumn(format="percent", min_value=0, max_value=1)},
        )
        if p["family"] == "Combined":
            legs = (
                holdings.assign(
                    leg2=holdings["Leg"].where(holdings["Leg"].isin(["Equity", "Crypto"]), "Single")
                )
                .groupby("leg2")["Weight"]
                .sum()
                .round(4)
            )
            st.caption(
                "Sleeve weights: " + " · ".join(f"{k} {legs.get(k, 0):.0%}" for k in legs.index)
            )

    st.subheader("Performance in the year")
    col_l, col_r = st.columns(2)
    g = ad.growth_series(fund, basis=basis)
    dd = ad.drawdown_series(fund, basis=basis)
    with col_l:
        fig, ax = as_.ft_figure((4.6, 3.0))
        ax.plot(g.index, g.values, color=as_.FT_PALETTE[0], lw=1.6)
        as_.ft_title(ax, "Growth of $1")
        ax.set_ylabel("Growth of $1")
        st.pyplot(fig)
    with col_r:
        fig, ax = as_.ft_figure((4.6, 3.0))
        ax.fill_between(dd.index, dd.values, 0, color=as_.FT_PALETTE[4], alpha=0.35)
        ax.plot(dd.index, dd.values, color=as_.FT_PALETTE[4], lw=1.2)
        as_.ft_title(ax, "Drawdown from peak")
        ax.set_ylabel("Drawdown")
        st.pyplot(fig)

    if p["family"] == "Combined" and p["method"] in ("Minimum Variance", "Maximum Sharpe"):
        img = ad.FIGURES / f"weights_over_time_combined_{p['method'].lower().replace(' ', '_')}.png"
        if img.exists():
            with centred_media():
                st.image(img, caption="Monthly rebalanced weights over time.")
    st.markdown(
        f"<div class='ever-muted' style='font-size:0.85rem'>{LIMITS}</div>", unsafe_allow_html=True
    )


def page_allocate() -> None:
    page_header("Allocate")
    st.markdown(
        "Build a blended portfolio from the systematic funds and see "
        "what the 2023 out-of-sample year would have looked like."
    )

    funds = ad.all_funds()
    selected = st.multiselect(
        "Choose funds (up to 5)",
        funds,
        default=funds[:3],
        max_selections=5,
        format_func=ad.dropdown_label,
    )
    if not selected:
        st.info("Select at least one fund to build a blend.")
        return

    basis = basis_control("alloc_basis")
    cols = st.columns(len(selected))
    weights = {}
    defaults = {
        f: 100 // len(selected) + (1 if i < 100 % len(selected) else 0)
        for i, f in enumerate(selected)
    }
    for col, fund in zip(cols, selected):
        with col:
            weights[fund] = st.number_input(
                f"Share of {ad.dropdown_label(fund)}",
                min_value=0,
                max_value=100,
                step=5,
                value=defaults[fund],
                key=f"alloc_{fund}",
            )

    total = sum(weights.values())
    if total == 100:
        st.success(f"Allocated: {total}% - well done.")
    else:
        st.warning(f"Allocated {total}% - shares are scaled to 100% in the blend below.")
    norm = {f: w / total for f, w in weights.items()}

    blended = ad.blend_funds(selected, norm, basis=basis)
    ppy = ad.annualising_weights(selected)
    stats = ad.metrics_from_series(blended, ppy)

    c_chart, c_table = st.columns([1.6, 1])
    with c_chart:
        fig, ax = as_.ft_figure((7.2, 3.4))
        g = blended.add(1).cumprod()
        ax.plot(g.index, g.values, color=as_.FT_PALETTE[0], lw=1.8)
        as_.ft_title(ax, f"Blended growth of $1 ({basis}, 2023)")
        ax.set_ylabel("Growth of $1")
        st.pyplot(fig)
    with c_table:
        alloc = pd.DataFrame(
            [
                {"Fund": ad.dropdown_label(f), "Share": v}
                for f, v in sorted(norm.items(), key=lambda kv: -kv[1])
            ]
        )
        st.dataframe(
            alloc,
            hide_index=True,
            width="stretch",
            column_config={"Share": cc.NumberColumn(format="percent", min_value=0, max_value=1)},
        )

    st.markdown("")
    st.markdown("**Blended portfolio, 2023 out-of-sample**")
    stat_row(stats)
    st.caption(
        f"Annualised with {ppy} trading days per year. Blended from "
        f"the funds' net-of-cost daily returns, capital-weighted."
    )
    st.markdown(
        f"<div class='ever-muted' style='font-size:0.85rem'>{LIMITS}</div>", unsafe_allow_html=True
    )


def page_market_insights() -> None:
    page_header("Market Insights")
    st.info(
        "**How to read this page.** Sentiment measures the *tone* of financial "
        "news headlines, not live market information. The scores cover the "
        "2020-2023 study window and fund results are evaluated out-of-sample "
        "on **2023 only**; sentiment does not predict returns with certainty."
    )

    st.subheader("What the news indicated")
    st.markdown(
        "Over the study window the tone of company news was mildly positive "
        "on average: a mean headline score of +0.11 under VADER and +0.07 "
        "under finVADER on a -1 (negative) to +1 (positive) scale, from "
        "146,836 headlines. VADER leaves about half of headlines neutral, "
        "while finVADER's finance lexicons reclassify many of those into "
        "signed readings - which is why the two models tell slightly "
        "different stories."
    )

    st.subheader("Sentiment across sectors")
    st.image(
        ad.FIGURES / "sentiment_sector_grid_finvader.png",
        width="stretch",
        caption="finVADER sentiment index by sector, 21-day average, 2020-2023.",
    )
    st.markdown(
        "News tone was most positive for **Technology** and **Healthcare** "
        "under finVADER (and **Utilities** and **Technology** under VADER). "
        "**Materials** was the weakest sector under both models - its average "
        "headline tone came closest to neutral."
    )

    st.subheader("How news changes a portfolio")
    st.markdown(
        "Each month, the baseline portfolio weights are tilted by the most "
        "recent **one-day-lagged** news score: assets with relatively "
        "positive news gain a little weight, funded by assets with less "
        "positive news, and the tilt is renormalised so the fund stays fully "
        "invested and long-only. The chart shows how much weight the tilt "
        "moves for the flagship equity capital-preservation fund."
    )
    with centred_media():
        st.image(
            ad.FIGURES / "fusion_weight_tilt_equity.png",
            caption=(
                "Mean change in portfolio weight from the sentiment tilt (Equity Minimum Variance)."
            ),
        )

    st.subheader("Did news sentiment add value?")
    st.markdown(
        "For the flagship capital-preservation fund, fusing news sentiment "
        "into the monthly weights raised the net out-of-sample Sharpe ratio "
        "from **0.49 to 0.66** with VADER, while finVADER slightly reduced it "
        "(0.49 to 0.45) in this one-year test. That is why the News-Aware "
        "Capital Preservation strategy uses **VADER** - not because VADER is "
        "generally better, but because only the VADER tilt added value in "
        "this year's out-of-sample evaluation."
    )
    with centred_media():
        st.image(
            ad.FIGURES / "fusion_sharpe_equity.png",
            caption=(
                "Baseline vs VADER vs finVADER fused Sharpe, 2023 out-of-sample (gross vs net)."
            ),
        )
    ba = ad.load_fusion_before_after()
    ba = ba[ba["basis"] == "net"].copy()
    ba["fund"] = ba["fund"].map(ad.dropdown_label)
    ba = ba.sort_values("delta_sharpe", ascending=False)
    st.dataframe(
        ba[["fund", "baseline_sharpe", "fused_sharpe", "delta_sharpe"]],
        hide_index=True,
        width="stretch",
        column_config={
            "baseline_sharpe": cc.NumberColumn(format="%.2f"),
            "fused_sharpe": cc.NumberColumn(format="%.2f"),
            "delta_sharpe": cc.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        "Net of costs, 2023 out-of-sample. A positive delta means the "
        "sentiment tilt added risk-adjusted value over the unfused baseline. "
        "The Combined rows equal the Equity rows here because the "
        "minimum-variance optimiser assigns ~0% to crypto, so the Combined "
        "fund is effectively the Equity fund."
    )

    st.subheader("Explore the research")
    with st.expander("VADER vs finVADER headline scores"):
        st.markdown(
            "**Key finding:** finVADER reduced the proportion of neutral "
            "headlines from 48.9% to 16.9%, showing that its finance "
            "lexicons identified sentiment that plain VADER did not capture. "
            "However, this additional sentiment did not produce better "
            "portfolio performance in the 2023 backtest."
        )
        with centred_media():
            st.image(
                ad.FIGURES / "sentiment_vader_vs_finvader.png",
                caption="VADER vs finVADER headline compound scores.",
            )
        st.caption(
            "The chart reports correlation between aggregated ticker-day "
            "scores, while the table reports correlation between individual "
            "headline scores. This explains the small difference between the "
            "two correlation values."
        )
        cmp = ad.load_sentiment_comparison().copy()
        _stat_labels = {
            "n_headlines": "Headlines analysed",
            "pct_neutral": "Neutral headlines",
            "pct_nonneutral": "Positive or negative headlines",
            "mean_compound": "Average sentiment score",
            "std_compound": "Sentiment score variation",
            "pearson_corr_vader_finvader": "Agreement between models",
            "vader_neutral_finvader_signed": "VADER-neutral headlines classified by finVADER",
            "sign_flip_between_models": "Opposite sentiment classifications",
        }
        _pct_stats = {
            "Neutral headlines",
            "Positive or negative headlines",
            "VADER-neutral headlines classified by finVADER",
            "Opposite sentiment classifications",
        }
        cmp = ad.load_sentiment_comparison()
        _rows = []
        for _, _r in cmp.iterrows():
            _stat = _stat_labels[_r["statistic"]]
            _vals = []
            for _col in ("vader", "finvader"):
                _v = float(_r[_col])
                if _stat in _pct_stats:
                    _vals.append(f"{_v * 100:.2f}%")
                elif _stat == "Headlines analysed":
                    _vals.append(f"{_v:,.0f}")
                else:
                    _vals.append(f"{_v:.4f}")
            _rows.append([_stat, *_vals])
        cmp = pd.DataFrame(_rows, columns=["statistic", "vader", "finvader"])
        st.dataframe(cmp, hide_index=True, width="stretch")
        st.caption(
            "Sentiment is lagged one trading day and neutral (0) on days "
            "with no headlines; indexes are equal-weighted across sectors."
        )
    with st.expander("Sentiment index over time, by sector (2020-2023)"):
        st.markdown(
            "**Key finding:** sector sentiment stayed positive for most of "
            "2020-2023, and the ten sectors moved broadly together, with "
            "Materials the least positive sector under both models."
        )
        c1, c2 = st.columns(2)
        with c1, centred_media(ratio=0.92):
            st.image(
                ad.FIGURES / "sentiment_sector_index_over_time_vader.png",
                caption="VADER sentiment index by sector over time.",
            )
        with c2, centred_media(ratio=0.92):
            st.image(
                ad.FIGURES / "sentiment_sector_index_over_time_finvader.png",
                caption="finVADER sentiment index by sector over time.",
            )
    with st.expander("How sensitive is the tilt to its strength?"):
        st.markdown(
            "**Key finding:** the stronger the VADER tilt, the higher the net "
            "Sharpe (0.49 at λ = 0 rising to 0.77 at λ = 1); the stronger the "
            "finVADER tilt, the lower it fell (0.42 at λ = 1). The app's "
            "λ = 0.5 sits in between and was fixed in advance, not tuned."
        )
        lam = ad.load_fusion_lambda()
        lam = lam[lam["family"].str.lower() == "equity"]
        fig, ax = as_.ft_figure((7.6, 3.6))
        for i, (model, grp) in enumerate(lam.groupby("model")):
            grp = grp.sort_values("lambda")
            ax.plot(
                grp["lambda"],
                grp["sharpe"],
                marker="o",
                ms=4,
                lw=1.6,
                color=as_.FT_PALETTE[i % len(as_.FT_PALETTE)],
                label=model,
            )
        ax.axvline(0.5, color=as_.FT_TEXT, lw=1.0, ls=(0, (4, 3)), alpha=0.8)
        ax.set_xlabel("Sentiment tilt strength λ")
        ax.set_ylabel("Sharpe (net, 2023)")
        as_.ft_title(ax, "Net Sharpe vs sentiment tilt strength")
        ax.legend(fontsize=8, frameon=False)
        with centred_media():
            st.pyplot(fig)
        st.caption(
            "λ = 0.5 is the predetermined tilt strength used in the headline "
            "fused funds; it was chosen as a fixed assumption, not tuned to "
            "this data."
        )
    with st.expander("Sector sentiment summary (table)"):
        st.markdown(
            "**Key finding:** news was positive on the large majority of "
            "trading days in every sector, and Materials was the most "
            "negative sector on average under both models."
        )
        summary = ad.load_sector_summary().rename(
            columns={
                "sector": "Sector",
                "n_days": "Days observed",
                "vader_mean": "VADER average",
                "vader_std": "VADER variation",
                "vader_pct_positive": "VADER positive days",
                "finvader_mean": "finVADER average",
                "finvader_std": "finVADER variation",
                "finvader_pct_positive": "finVADER positive days",
            }
        )
        st.dataframe(
            summary,
            hide_index=True,
            width="stretch",
            column_config={
                "VADER positive days": cc.NumberColumn(format="percent"),
                "finVADER positive days": cc.NumberColumn(format="percent"),
            },
        )


def page_methodology() -> None:
    page_header("Methodology")
    st.markdown(
        "Every equation behind the figures in this app, in one place. All "
        "values are computed offline in `scripts/run_part_b.py` from the "
        "course data bundle and loaded into the app as precomputed artifacts."
    )

    st.subheader("1. Daily returns and calendars")
    st.markdown(
        "Each asset's simple daily return is computed from its adjusted close "
        "price, within its own panel, before any cross-asset merge:"
    )
    st.latex(r"r_{i,t} = \frac{P^{adj}_{i,t}}{P^{adj}_{i,t-1}} - 1")
    st.markdown(
        "Returns are annualised with **252** trading days a year for "
        "equity/combined funds and **365** for crypto-only funds (crypto "
        "trades every calendar day). The combined panel left-merges crypto "
        "returns onto the equity trading calendar, so weekend-only crypto "
        "moves are excluded by construction. The sample runs "
        "**2020-01-01 to 2023-12-31** (crypto is capped at 2023-12-31 per the "
        "data guide)."
    )

    st.subheader("2. Walk-forward backtest (no look-ahead)")
    st.markdown(
        "- Initial estimation window: **2020-01-01 to 2022-12-31**.\n"
        "- Out-of-sample period: **2023** - first live backtest date "
        "**2023-01-03** for equity/combined (250 trading days) and "
        "**2023-01-01** for crypto (365 days).\n"
        "- Weights are re-estimated **monthly** on the first trading day of "
        "each month (plus the first OOS date), using an **expanding window** "
        "of all data strictly before the rebalance date:"
    )
    st.latex(
        r"w_t = \text{solve}(\, \mathcal{W}_t\, ),\qquad "
        r"\mathcal{W}_t = \{\, r_{i,s} : s < t\,\}"
    )
    st.markdown(
        "Weights are held flat between rebalances. At rebalance date $t$ only "
        "returns with $s<t$ are used, so no future information ever enters a "
        "weight."
    )

    st.subheader("3. Portfolio optimisation methods")
    st.markdown(
        "All methods are **long-only** ($0 \\le w_i \\le 1$) and **fully "
        "invested** ($\\sum_i w_i = 1$). $\\Sigma$ is the sample covariance "
        "and $\\mu$ the sample mean of daily returns in the training window."
    )
    st.markdown("**Equal Weight** - every asset an equal share:")
    st.latex(r"w_i = \frac{1}{N}")
    st.markdown("**Minimum Variance** - smallest predicted volatility:")
    st.latex(r"\min_{w}\; w^{\top}\Sigma\, w \quad \text{s.t. } \sum_i w_i=1,\; w\ge 0")
    st.markdown("**Maximum Sharpe** - best return per unit of risk ($r_f = 0$):")
    st.latex(
        r"\max_{w}\; \frac{w^{\top}\mu - r_f}{\sqrt{w^{\top}\Sigma\, w}} "
        r"\quad \text{s.t. } \sum_i w_i=1,\; w\ge 0"
    )
    st.markdown(
        "**Risk Parity** - every asset contributes an equal share of portfolio "
        "risk (convex log-barrier form of Spinu, 2013, normalised afterwards):"
    )
    st.latex(
        r"\min_{w}\; \frac{1}{2} w^{\top}\Sigma\, w - \frac{1}{N}\sum_{i=1}^{N} "
        r"\ln(w_i) \quad \text{then } w \leftarrow w / \sum_i w_i"
    )
    st.markdown(
        "**Minimum CVaR** (extension, $\\alpha = 0.95$) - minimises the "
        "expected loss in the worst 5% of training days via the "
        "Rockafellar-Uryasev linear programme over the $T$ historical return "
        "scenarios $R_t$:"
    )
    st.latex(
        r"\min_{w,\beta,z}\; \beta + \frac{1}{(1-\alpha)\,T}\sum_{t=1}^{T} z_t "
        r"\quad \text{s.t. } z_t \ge -w^{\top}R_t - \beta,\; z_t\ge 0,\; "
        r"\sum_i w_i=1,\; 0\le w \le w^{max}"
    )

    st.subheader("4. Performance metrics")
    st.markdown("Growth of $1 invested:")
    st.latex(r"W_T = \prod_{t=1}^{T} \left(1 + r_t\right)")
    st.markdown("Annualised return and annualised volatility (ppy = 252 or 365):")
    st.latex(
        r"\bar r_{ann} = \left(\prod_{t=1}^{T} (1+r_t)\right)^{ppy/T} - 1,"
        r"\qquad \sigma_{ann} = \sigma(r) \cdot \sqrt{ppy}"
    )
    st.markdown("Sharpe ratio with $r_f = 0$, and maximum drawdown:")
    st.latex(
        r"SR = \frac{\bar r_{ann}}{\sigma_{ann}},"
        r"\qquad MDD = \min_{t} \left( \frac{W_t}{\max_{s\le t} W_s} - 1 \right)"
    )

    st.subheader("5. Transaction costs (net vs gross)")
    st.markdown("One-way portfolio turnover and the 5 bps cost, charged on rebalance days:")
    st.latex(
        r"TO_t = \frac{1}{2}\sum_{i} \left| w_{i,t} - w_{i,t-1}\right|,"
        r"\qquad c_t = 0.0005 \cdot TO_t"
    )
    st.latex(r"r^{net}_t = (1 + r^{gross}_t)(1 - c_t) - 1")
    st.markdown(
        "The one-off initial allocation trade is excluded so every fund is "
        "charged for the same thing (rebalancing)."
    )

    st.subheader("6. Sentiment scoring and the sector index")
    st.markdown(
        "Every headline is scored with **plain VADER** and with **finVADER** "
        "(VADER plus the SentiBignomics$\\times 0.1$ and Henry finance "
        "lexicons), producing a compound score in $[-1, 1]$. A ticker-day "
        "score is the mean of its headlines' compounds, and ticker-days with "
        "no headlines are treated as neutral 0:"
    )
    st.latex(
        r"s_{i,t} = \frac{1}{H_{i,t}} \sum_{h=1}^{H_{i,t}} compound_h"
        r"\qquad (\text{no headlines} \Rightarrow s_{i,t}=0)"
    )
    st.markdown(
        "The equal-weight sector index averages the constituent tickers, "
        "then lags the whole series one trading day so day $t$ uses only news "
        "aligned to day $t-1$ or earlier:"
    )
    st.latex(
        r"S_{k,t} = \frac{1}{N_k}\sum_{i \in k} s_{i,t}, \qquad "
        r"S^{lag}_{k,t} = S_{k,\,t-1}"
    )
    st.markdown(
        "An alternative **coverage-weighted** index (innovation) weights each "
        "ticker-day reading by the number of headlines behind it, "
        "$n_{i,t}$, giving no-headline ticker-days zero weight:"
    )
    st.latex(
        r"S^{cov}_{k,t} = \frac{\sum_{i \in k} n_{i,t}\, s_{i,t}}"
        r"{\sum_{i \in k} n_{i,t}}"
    )
    st.markdown(
        "The sector grid figure plots the **21-day rolling average** of the "
        "lagged index: $\\bar{S}_{k,t} = \\frac{1}{21}\\sum_{j=0}^{20} "
        "S^{lag}_{k,t-j}$."
    )

    st.subheader("7. Sentiment fusion (the News-Aware funds)")
    st.markdown(
        "Fused weights tilt the baseline monthly weights by a "
        "cross-sectional standardised, one-day-lagged sentiment signal "
        "$z$, scaled by the tilt strength $\\lambda = 0.5$ "
        "(**predetermined assumption, not fitted to the data**):"
    )
    st.latex(
        r"z_{i,t} = \frac{s_{i,t} - \mathrm{mean}_k(s_t)}{\mathrm{std}_k(s_t)},"
        r"\qquad w'_{i,t} = \max\left(0,\; w_{i,t}\,(1 + \lambda\, z_{i,t})\right),"
        r"\qquad w''_{i,t} = \frac{w'_{i,t}}{\sum_j w'_{j,t}}"
    )
    st.markdown(
        "The tilt is mean-zero per date (high-sentiment names gain weight "
        "funded by low-sentiment names) and is applied to equity assets only; "
        "crypto and any asset without news get $z=0$. The fused fund's "
        "returns are recomputed from the fused weights: "
        "$r^{fused}_t = \\sum_i w''_{i,t}\\, r_{i,t}$."
    )

    st.subheader("8. Blended allocation (Allocate page)")
    st.markdown("A custom blend is a capital-weighted sum of fund returns on their common dates:")
    st.latex(r"r^{blend}_t = \sum_f w_f\, r^{f}_{t}, \qquad \sum_f w_f = 1")
    st.markdown(
        "Blend metrics use **365** days per year when at least half the "
        "chosen funds are crypto-only, otherwise **252**."
    )

    st.subheader("Key assumptions")
    st.markdown(
        "- Risk-free rate **$r_f = 0$** (no risk-free series is provided).\n"
        "- Long-only, fully invested, monthly rebalancing, weights held flat "
        "between rebalances.\n"
        "- Transaction costs: a flat **5 bps** per unit of one-way turnover.\n"
        "- Expanding estimation window (later rebalances see more data).\n"
        "- One out-of-sample year (2023) is a thin evaluation slice."
    )


def page_about() -> None:
    page_header("About the data")
    st.markdown("**What was computed** (all offline, in `results/`)")
    st.markdown(
        "- **Universes:** equity (50 US large-caps across 10 sectors) and crypto"
        " (top assets), plus diversified funds holding an equity + crypto mix.\n"
        "- **Strategies:** Equal Weight, Minimum Variance, Maximum Sharpe, "
        "Risk Parity, and a Minimum CVaR extension - optimised on "
        "2020-2022, evaluated out-of-sample on 2023.\n"
        "- **Investor-friendly names:** Diversified Core = Equal Weight, "
        "Capital Preservation = Minimum Variance, Risk-Adjusted Growth = "
        "Maximum Sharpe, Balanced Risk = Risk Parity, Downside Protection = "
        "Minimum CVaR, News-Aware Capital Preservation = fused sentiment "
        "funds.\n"
        "- **Costs:** 5bps one-way on rebalance trades, excluded from the "
        "initial allocation; net-of-cost series underpin every fund story.\n"
        "- **Sentiment:** daily headline scores from VADER and finVADER "
        "(VADER + finance lexicons), lagged one day, fused into fund weights "
        "via a cross-sectional z-score tilt scaled by λ = 0.5.\n"
        "- **Report charts** share the same FT house style as this app."
    )
    st.markdown("**How to read the metrics**")
    st.markdown(
        "- Returns are annualised with 252 trading days (equity/combined) or "
        "365 (crypto); Sharpe uses rf = 0.\n"
        "- Max drawdown is peak-to-trough, daily, over 2023.\n"
        "- 'Gross' excludes trading costs; 'Net' is what an investor keeps."
    )
    st.markdown("**Limitations**")
    st.markdown(
        "- One out-of-sample year is a thin slice; results are illustrative "
        "of the method, not a prediction.\n"
        "- Headline sentiment is noisy and forward-looking documents are "
        "excluded; VADER is known to neutralise finance jargon (finVADER "
        "helps).\n"
        "- Sample covariance and a flat 5bps cost are simplifying "
        "assumptions for coursework.\n"
        "- This is an educational decision-support tool, **not** investment "
        "advice."
    )


PAGE_FNS = {
    "Home": page_home,
    "Compare funds": page_compare,
    "Fund fact sheet": page_fact_sheet,
    "Allocate": page_allocate,
    "Market Insights": page_market_insights,
    "Methodology": page_methodology,
    "About the data": page_about,
}


def main() -> None:
    st.set_page_config(page_title="EverVest", layout="wide")
    as_.inject_css()
    page = sidebar_nav()
    PAGE_FNS[page]()


main()
