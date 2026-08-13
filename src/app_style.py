"""EverVest app design system (FT-inspired, mirrors src/plotting.py tokens).

The app is a thin reader over results/ and its charts either reuse the
precomputed FT-style PNGs or are drawn here with the same tokens, so the app
and the report share one visual language: cream background, dark text, faint
horizontal gridlines, and the burgundy/blue/teal/orange/purple palette.
"""

import matplotlib as mpl

FT_BG = "#F7F1E8"
FT_BG_2 = "#EFE7D9"
FT_TEXT = "#333333"
FT_GRID = "#D9D2C7"
FT_PALETTE = ["#7B1E3D", "#0F5499", "#1E7A6E", "#C9601C", "#5B3A8E"]

FONT_HEAD = "Georgia, 'Times New Roman', serif"
FONT_BODY = "-apple-system, 'Segoe UI', Arial, sans-serif"

_CSS = """
<style>
:root {
  --ever-bg: #F7F1E8;
  --ever-bg2: #EFE7D9;
  --ever-card: #FFFFFF;
  --ever-line: #E3DACB;
  --ever-text: #333333;
  --ever-grid: #D9D2C7;
  --ever-burgundy: #7B1E3D;
  --ever-blue: #0F5499;
  --ever-teal: #1E7A6E;
  --ever-orange: #C9601C;
}
html, body, [class*="css"] {
  font-family: __FONT_BODY__;
  color: var(--ever-text);
}
.stApp { background-color: var(--ever-bg); }
h1, h2, h3, h4 {
  font-family: __FONT_HEAD__;
  color: var(--ever-burgundy);
  letter-spacing: -0.01em;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background-color: var(--ever-bg2); }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--ever-burgundy); }
.stButton > button {
  background-color: var(--ever-burgundy);
  color: var(--ever-bg);
  border: none;
  border-radius: 4px;
  padding: 0.6rem 1.5rem;
  font-weight: 600;
}
.stButton > button:hover {
  background-color: #5e162d;
  color: var(--ever-bg);
  border: none;
}
.ever-hero {
  background: var(--ever-card);
  border: 1px solid var(--ever-line);
  border-left: 6px solid var(--ever-burgundy);
  border-radius: 6px;
  padding: 1.6rem 2rem;
  box-shadow: 0 2px 6px rgba(51, 51, 51, 0.08);
}
.ever-hero .kicker {
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ever-orange);
}
.ever-hero h1 { margin: 0.2rem 0 0.4rem 0; font-size: 2.3rem; }
.ever-hero p { margin: 0.3rem 0 0; font-size: 1.05rem; line-height: 1.5; }
.ever-card {
  background: var(--ever-card);
  border: 1px solid var(--ever-line);
  border-radius: 6px;
  padding: 1rem 1.2rem;
  box-shadow: 0 2px 6px rgba(51, 51, 51, 0.06);
}
.ever-card h4 { margin: 0 0 0.4rem 0; }
.ever-stat {
  background: var(--ever-card);
  border: 1px solid var(--ever-line);
  border-top: 3px solid var(--ever-burgundy);
  border-radius: 6px;
  padding: 0.9rem 1.1rem;
  text-align: center;
}
.ever-stat .k { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: #7A7264; }
.ever-stat .v { font-size: 1.5rem; font-weight: 700; color: var(--ever-burgundy); }
.ever-stat .n { font-size: 0.8rem; color: #7A7264; }
.ever-muted { color: #7A7264; }
.ever-rule { border: none; border-top: 1px solid var(--ever-grid); margin: 1.4rem 0; }
[data-testid="stMetric"] {
  background: var(--ever-card);
  border: 1px solid var(--ever-line);
  border-radius: 6px;
  padding: 0.7rem 1rem;
}
[data-testid="stMetricValue"] { color: var(--ever-burgundy); }
.ever-group {
  font-family: __FONT_HEAD__;
  font-weight: 700;
  font-size: 1.15rem;
  color: var(--ever-burgundy);
  margin: 1.4rem 0 0.6rem;
  padding-bottom: 0.25rem;
  border-bottom: 1px solid var(--ever-grid);
}
.ever-fund {
  background: var(--ever-card);
  border: 1px solid var(--ever-line);
  border-left: 4px solid var(--ever-burgundy);
  border-radius: 6px;
  padding: 0.9rem 1.15rem;
  margin-bottom: 0.8rem;
  box-shadow: 0 2px 6px rgba(51, 51, 51, 0.06);
}
.ever-fund .fname {
  font-family: __FONT_HEAD__;
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--ever-burgundy);
}
.ever-fund .fsub { font-size: 0.8rem; color: #7a7264; margin: 0.1rem 0 0.35rem; }
.ever-fund .fdesc { font-size: 0.88rem; line-height: 1.45; color: var(--ever-text); }
.ever-fund .funi { font-size: 0.82rem; color: #7a7264; margin-top: 0.4rem; }
</style>
""".replace("__FONT_HEAD__", FONT_HEAD).replace("__FONT_BODY__", FONT_BODY)


def inject_css() -> None:
    """Inject the EverVest design system CSS once."""
    import streamlit as st

    st.markdown(_CSS, unsafe_allow_html=True)


def stat_html(key: str, value: str, note: str = "") -> str:
    """Render one stat card (label / value / optional note)."""
    note_html = f'<div class="n">{note}</div>' if note else ""
    return (
        f'<div class="ever-stat"><div class="k">{key}</div>'
        f'<div class="v">{value}</div>{note_html}</div>'
    )


def card_html(title: str, body: str) -> str:
    """Render one titled content card."""
    return f'<div class="ever-card"><h4>{title}</h4>{body}</div>'


def hero_html(kicker: str, title: str, body: str) -> str:
    """Render the hero banner (kicker / headline / value proposition)."""
    return (
        f'<div class="ever-hero"><div class="kicker">{kicker}</div>'
        f"<h1>{title}</h1><p>{body}</p></div>"
    )


def group_heading_html(title: str) -> str:
    """Render a catalogue section heading."""
    return f'<div class="ever-group">{title}</div>'


def fund_card_html(name: str, subtitle: str, desc: str, footer: str = "") -> str:
    """Render one catalogue fund card (friendly name / subtitle / blurb)."""
    desc_html = f'<div class="fdesc">{desc}</div>' if desc else ""
    foot_html = f'<div class="funi">{footer}</div>' if footer else ""
    return (
        f'<div class="ever-fund"><div class="fname">{name}</div>'
        f'<div class="fsub">{subtitle}</div>{desc_html}{foot_html}</div>'
    )


def style_ax(ax) -> None:
    """Apply FT chart styling to a fresh matplotlib axis."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, colors=FT_TEXT, labelsize=9)
    ax.yaxis.grid(True, color=FT_GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.set_facecolor(FT_BG)


def ft_figure(figsize=(8.4, 4.2)):
    """New figure + styled axis on the FT cream background."""
    import matplotlib.pyplot as plt

    mpl.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(FT_BG)
    style_ax(ax)
    return fig, ax


def ft_title(ax, title: str, fontsize: float = 12) -> None:
    """Bold left-aligned FT title."""
    ax.set_title(title, loc="left", fontsize=fontsize, fontweight="bold", color=FT_TEXT)
