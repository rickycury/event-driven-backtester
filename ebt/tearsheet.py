"""Four-panel tearsheet: equity + drawdowns, rolling Sharpe,
trade-level PnL distribution, gross exposure."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import drawdown_series, rolling_sharpe, summary

plt.style.use("seaborn-v0_8-whitegrid")


def trade_pnls(trades: pd.DataFrame) -> pd.Series:
    """FIFO round-trip PnL per closed trade; handles longs and shorts.

    Inventory holds signed lots. A trade in the same direction adds a lot;
    an opposing trade closes lots FIFO (realizing PnL) and any excess flips
    into a new lot in the opposite direction.
    """
    pnls, inventory = [], []  # lots: [signed_qty, price]
    for _, t in trades.sort_values("timestamp").iterrows():
        q, p = int(t["quantity"]), float(t["price"])
        while q != 0:
            if not inventory or (inventory[0][0] > 0) == (q > 0):
                inventory.append([q, p])
                q = 0
            else:
                lot = inventory[0]
                take = min(abs(q), abs(lot[0]))
                direction = 1 if lot[0] > 0 else -1   # long lot closed by sell
                pnls.append(take * direction * (p - lot[1]))
                lot[0] -= direction * take
                q += direction * take
                if lot[0] == 0:
                    inventory.pop(0)
    return pd.Series(pnls, name="trade_pnl", dtype=float)


def make_tearsheet(results: pd.DataFrame, trades: pd.DataFrame,
                   costs: dict, path: str, title: str) -> pd.Series:
    eq, rets = results["equity"], results["returns"]
    stats = summary(results, trades, costs)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    # 1. equity with drawdown shading
    ax = axes[0, 0]
    ax.plot(eq.index, eq.values, color="#1f4e79", lw=1.4)
    dd = drawdown_series(eq)
    ax2 = ax.twinx()
    ax2.fill_between(dd.index, dd.values * 100, 0, color="#c0392b", alpha=0.25)
    ax2.set_ylabel("Drawdown (%)", color="#c0392b", fontsize=10)
    ax2.set_ylim(dd.min() * 100 * 3, 0)
    ax.set_title(f"Equity Curve  |  CAGR {stats['CAGR']:.1%}, "
                 f"MaxDD {stats['Max Drawdown']:.1%}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Equity ($)", fontsize=10)
    ax.yaxis.set_major_formatter(lambda x, _: f"${x/1e6:.2f}M")

    # 2. rolling Sharpe
    ax = axes[0, 1]
    rs = rolling_sharpe(rets)
    ax.plot(rs.index, rs.values, color="#1f4e79", lw=1.2)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axhline(stats["Sharpe"], color="#c0392b", ls="--", lw=1,
               label=f"Full-period Sharpe {stats['Sharpe']:.2f}")
    ax.set_title("Rolling 6-Month Sharpe (OOS)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # 3. trade PnL distribution
    ax = axes[1, 0]
    pnl = trade_pnls(trades)
    if not pnl.empty:
        ax.hist(pnl, bins=30, color="#1f4e79", alpha=0.85, edgecolor="white")
        ax.axvline(0, color="grey", lw=0.8)
        ax.axvline(pnl.mean(), color="#c0392b", ls="--", lw=1,
                   label=f"Mean ${pnl.mean():,.0f} | Win rate {(pnl > 0).mean():.0%}")
        ax.legend(fontsize=9)
    ax.set_title(f"Round-Trip Trade PnL ({len(pnl)} closed trades)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("PnL ($)", fontsize=10)
    ax.xaxis.set_major_formatter(lambda x, _: f"${x/1e3:.0f}K")

    # 4. gross exposure
    ax = axes[1, 1]
    exp = results["gross_exposure"] * 100
    ax.fill_between(exp.index, exp.values, color="#1f4e79", alpha=0.5)
    ax.set_title("Gross Exposure Over Time", fontsize=12, fontweight="bold")
    ax.set_ylabel("% of Equity", fontsize=10)
    ax.set_ylim(0, max(110, exp.max() * 1.05))

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stats
