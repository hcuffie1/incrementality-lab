"""Static figures for the report. One axis per chart, thin marks, recessive grid."""
from __future__ import annotations

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#d6d5d0")
    ax.grid(axis="y", color="#e7e6e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def qini_plot(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    colors = [BLUE, ORANGE, AQUA]
    last = None
    for (name, (frac, q)), c in zip(curves.items(), colors):
        ax.plot(frac * 100, q * 1000, color=c, linewidth=2, label=name)
        last = (frac, q)
    frac, q = last
    ax.plot([0, 100], [0, q[-1] * 1000], color=GRAY, linewidth=1.5, linestyle="--", label="Random targeting")
    _style(ax, title, "Share of customers targeted, ranked by predicted uplift (%)",
           "Incremental visits per 1,000 customers")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def decile_plot(rows: list[dict], path: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    d = np.array([r["decile"] for r in rows])
    obs = np.array([r["observed"] for r in rows]) * 100
    pred = np.array([r["predicted"] for r in rows]) * 100
    ax.bar(d - 0.2, pred, width=0.38, color=BLUE, label="Predicted uplift")
    ax.bar(d + 0.2, obs, width=0.38, color=ORANGE, label="Observed (treated minus control)")
    ax.axhline(0, color="#bdbcb6", linewidth=1)
    ax.set_xticks(d)
    _style(ax, title, "Predicted-uplift decile (1 = highest)", "Visit rate lift (percentage points)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def policy_plot(rows: list[dict], path: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    x = np.array([r["share_emailed"] for r in rows]) * 100
    y = np.array([r["net_profit_per_1000_customers"] for r in rows])
    lo = np.array([r["profit_low"] for r in rows])
    hi = np.array([r["profit_high"] for r in rows])
    ax.fill_between(x, lo, hi, color=BLUE, alpha=0.15, linewidth=0, label="95% interval")
    ax.plot(x, y, color=BLUE, linewidth=2, marker="o", markersize=5, label="Net profit")
    ax.axhline(0, color="#bdbcb6", linewidth=1)
    best = int(np.argmax(y))
    ax.annotate(f"Best: email top {x[best]:.0f}%", (x[best], y[best]), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9, color=INK)
    _style(ax, title, "Share of customers emailed, ranked by predicted uplift (%)",
           "Net profit per 1,000 customers ($)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def event_study_plot(es, path: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.axvspan(-0.5, es["relative_week"].max() + 0.5, color="#f0efec", linewidth=0)
    ax.plot(es["relative_week"], es["gap"] * 100, color=BLUE, linewidth=2, marker="o", markersize=5)
    ax.axhline(0, color="#bdbcb6", linewidth=1)
    ax.text(0, ax.get_ylim()[1] * 0.92, "  Launch", color=INK2, fontsize=9)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    _style(ax, title, "Weeks relative to launch", "Treated minus control (% of sales, vs pre-period average)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
