"""
Financial-Times-style plotting helpers, matched to the course reference
slide ("The Same Series, Cleaned Up"):
- Off-white/cream background (#F7F1E8), not pure white.
- NO axis box at all - no spines, no tick marks, just tick labels.
- Faint horizontal gridlines only (no vertical gridlines).
- Bold title, left-aligned, sitting above the axes.
- The line's LAST value is labelled directly on the chart, in the line's
  own colour, instead of a legend - matches "the final value labelled".
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

FT_BG = "#F7F1E8"
FT_TEXT = "#333333"
FT_GRID = "#D9D2C7"
FT_PALETTE = ["#7B1E3D", "#0F5499", "#1E7A6E", "#C9601C", "#5B3A8E"]


def ft_figure(figsize=(8, 4.2)):
    """Figure + axis with the FT background and no visible axis box."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(FT_BG)
    ax.set_facecolor(FT_BG)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="both", length=0, colors=FT_TEXT, labelsize=9)
    ax.yaxis.grid(True, color=FT_GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    return fig, ax


def ft_title(ax, title: str, pad: float = 12):
    """Bold, left-aligned title (FT style). `pad` gives the gap in points
    between the axes and the title - raise it when a subtitle is added so
    the two lines never touch."""
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=FT_TEXT, pad=pad)


def ft_line(ax, x, y, label_value: str, color=None, index=0):
    """Plot a single line and label its LAST point with a value, in the
    line's own colour, instead of a legend entry."""
    color = color or FT_PALETTE[index % len(FT_PALETTE)]
    ax.plot(x, y, color=color, linewidth=1.5)
    ax.annotate(label_value, xy=(x[-1], y[-1]), xytext=(6, 0),
                textcoords="offset points", color=color,
                fontsize=9.5, fontweight="bold", va="center")


def _declutter_y(values, min_gap):
    """Given target y-positions, push overlapping ones apart (ascending
    pass) so no two labels sit closer than min_gap. Returns adjusted
    y-positions in the same order as `values`."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [values[i] for i in order]
    for k in range(1, len(adjusted)):
        if adjusted[k] - adjusted[k - 1] < min_gap:
            adjusted[k] = adjusted[k - 1] + min_gap
    out = [0.0] * len(values)
    for pos, i in enumerate(order):
        out[i] = adjusted[pos]
    return out


def ft_multi_line_end_labels(ax, series_dict: dict, max_end_labels: int = 4):
    """Plot several (label -> pandas Series) lines. If there are few
    enough lines (<= max_end_labels), each gets a name label at its own
    line-end, decluttered so text never overlaps (nudged apart with a
    thin leader line back to the real last point). If there are MORE
    lines than that, end-labelling gets crowded no matter how much you
    declutter it - so instead this falls back to a small, unobtrusive
    legend in the top-left, still borderless/FT-ish, rather than forcing
    unreadable stacked labels."""
    import matplotlib.patheffects as pe

    items = list(series_dict.items())
    colors = {}
    for i, (label, s) in enumerate(items):
        color = FT_PALETTE[i % len(FT_PALETTE)]
        colors[label] = color
        ax.plot(s.index, s.values, color=color, linewidth=1.5, label=label)

    if len(items) > max_end_labels:
        leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5,
                         handlelength=1.4, labelspacing=0.3)
        for text in leg.get_texts():
            text.set_color(FT_TEXT)
        return

    ax.figure.canvas.draw()
    ymin, ymax = ax.get_ylim()
    min_gap = (ymax - ymin) * 0.07  # widened from 0.045 - 4 crowded lines need more room

    end_x = [s.index[-1] for _, s in items]
    end_y = [s.values[-1] for _, s in items]
    adj_y = _declutter_y(end_y, min_gap)

    halo = [pe.withStroke(linewidth=3, foreground=FT_BG)]
    for (label, s), x, y_real, y_label in zip(items, end_x, end_y, adj_y):
        color = colors[label]
        if abs(y_label - y_real) > min_gap * 0.2:
            ax.plot([x, x], [y_real, y_label], color=color, linewidth=0.7, alpha=0.6)
        txt = ax.annotate(label, xy=(x, y_label), xytext=(6, 0),
                           textcoords="offset points", color=color,
                           fontsize=9, fontweight="bold", va="center")
        txt.set_path_effects(halo)  # cream halo so text stays legible over crossing lines


def ft_bar(ax, labels, values, color=None):
    """Horizontal bar chart in FT style (used for the top-terms exhibit)."""
    color = color or FT_PALETTE[0]
    ax.barh(labels, values, color=color)
    ax.xaxis.grid(True, color=FT_GRID, linewidth=0.8)
    ax.yaxis.grid(False)


def ft_grouped_bar(ax, categories, series_dict: dict, bar_height=0.35):
    """Horizontal grouped bar chart - e.g. avg_positive_words vs
    avg_negative_words side by side per sector, so relative magnitude
    across sectors is directly comparable at a glance."""
    n_groups = len(series_dict)
    y = np.arange(len(categories))
    for i, (label, values) in enumerate(series_dict.items()):
        color = FT_PALETTE[i % len(FT_PALETTE)]
        offset = (i - (n_groups - 1) / 2) * bar_height
        ax.barh(y + offset, values, height=bar_height, color=color, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.xaxis.grid(True, color=FT_GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    for text in leg.get_texts():
        text.set_color(FT_TEXT)


def ft_heatmap(fig, ax, matrix, cmap="YlGnBu", cbar_label="count", log_scale=True):
    """FT-consistent heatmap: cream figure background, no axis box, bold
    left-aligned title expected to be set separately via ft_title().
    matrix: 2D array-like with .index (rows) and .columns (cols), e.g. a
    pandas DataFrame from features.news_coverage_matrix().

    log_scale=True (default): colour on log(count + 1) instead of raw
    count. Without this, one outlier ticker/month (e.g. a stock caught in
    a hype cycle) can dominate the scale so hard that every other cell
    looks the same pale colour, hiding the coverage-gap pattern the
    heatmap exists to show. Log scaling keeps the outlier visible while
    restoring contrast among the rest. +1 avoids log(0) for genuinely
    zero-headline ticker-months.
    """
    import matplotlib.colors as mcolors

    if log_scale:
        display_vals = np.log1p(matrix.values)
        norm = mcolors.Normalize(vmin=0, vmax=display_vals.max())
        im = ax.imshow(display_vals, aspect="auto", cmap=cmap, norm=norm)
    else:
        im = ax.imshow(matrix.values, aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7, color=FT_TEXT)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=7, color=FT_TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=7, colors=FT_TEXT)
    if log_scale:
        # show real headline counts on the colourbar, not log-transformed numbers
        raw_ticks = [0, 5, 20, 50, 100, 200, 400, 800]
        raw_ticks = [t for t in raw_ticks if t <= matrix.values.max()]
        cbar.set_ticks(np.log1p(raw_ticks))
        cbar.set_ticklabels([str(t) for t in raw_ticks])
    cbar.set_label(cbar_label + (" (log scale)" if log_scale else ""), fontsize=8, color=FT_TEXT)
    return im
