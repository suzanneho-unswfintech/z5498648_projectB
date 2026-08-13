"""Report figure: VADER vs finVADER tilt-sensitivity (lambda sweep).

Reads the precomputed results/tables/fusion_lambda_sensitivity.csv ONLY
(no backtest, no re-computation), and redraws the existing app chart as a
publication-quality FT-style PNG for the report.

The chart follows the other report exhibits (src/plotting.py): cream
background, no axis box, faint horizontal gridlines, bold left-aligned
title, and the FT palette. lambda = 0 is the unfused baseline (both
lexicons identical by construction); lambda = 0.5 marks the predetermined
tilt strength used for the headline fused funds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import plotting as ft

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "results" / "tables" / "fusion_lambda_sensitivity.csv"
FIG_DIR = ROOT / "results" / "figures"

TITLE = "Sensitivity of Risk-Adjusted Performance to Sentiment Tilt Strength"
X_LABEL = "Sentiment Tilt Strength (\u03bb)"
Y_LABEL = "Net Out-of-Sample Sharpe Ratio (2023)"
SELECTED = 0.5
MODEL_COLORS = {"VADER": ft.FT_PALETTE[0], "finVADER": ft.FT_PALETTE[1]}


def main() -> None:
    lam = pd.read_csv(TABLE)
    lam = lam[lam["basis"] == "net"].copy()
    lam = lam[lam["family"].str.lower() == "equity"].copy()
    lam = lam.sort_values(["model", "lambda"])

    fig, ax = ft.ft_figure(figsize=(8.6, 4.9))
    for model, grp in lam.groupby("model", sort=True):
        ax.plot(
            grp["lambda"],
            grp["sharpe"],
            marker="o",
            ms=5.5,
            lw=1.8,
            color=MODEL_COLORS[model],
            label=model,
        )

    ax.axvline(SELECTED, color=ft.FT_TEXT, lw=1.0, ls=(0, (4, 3)), alpha=0.8)
    ax.text(
        SELECTED + 0.015,
        0.965,
        "Selected tilt strength",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8.5,
        color=ft.FT_TEXT,
    )

    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo - (yhi - ylo) * 0.16, yhi)
    ax.annotate(
        "Baseline: no sentiment tilt",
        xy=(0.0, lam["sharpe"].min()),
        xytext=(0, -13),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8.5,
        color=ft.FT_TEXT,
    )

    ax.set_xlabel(X_LABEL, color=ft.FT_TEXT, fontsize=9.5)
    ax.set_ylabel(Y_LABEL, color=ft.FT_TEXT, fontsize=9.5)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.03, 1.03)
    ft.ft_title(ax, TITLE)

    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(ft.FT_TEXT)

    ax.figure.tight_layout()
    fig.savefig(
        FIG_DIR / "fusion_lambda_sensitivity_equity.png", dpi=300, facecolor=fig.get_facecolor()
    )
    plt.close(fig)

    pivot = lam.pivot_table(index="lambda", columns="model", values="sharpe")
    pivot = pivot.round(4)
    pivot.to_csv(ROOT / "results" / "tables" / "fusion_lambda_sensitivity_plotted.csv")
    print("Plotted values (net Sharpe, equity family):")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
