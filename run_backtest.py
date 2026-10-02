"""End-to-end run: data -> walk-forward -> frictions -> tearsheet.

Default uses the built-in synthetic generator so the whole pipeline is
reproducible offline. Swap in real data with:

    data = ebt.data.from_yahoo(["SPY"], "2016-01-01", "2026-01-01")
    # or: data = ebt.data.from_csv("spy.csv", "SPY")

and pull an actual financing rate from FRED (series DFF or SOFR) for
FrictionConfig.borrow_rate_annual instead of the hardcoded proxy.
"""
import pandas as pd

from ebt import data as datamod
from ebt.metrics import deflated_sharpe
from ebt.execution import FrictionConfig
from ebt.tearsheet import make_tearsheet
from ebt.validation import walk_forward

SYMBOL = "SYN"

def main():
    data = datamod.synthetic(SYMBOL, n_days=2520, seed=7)

    cfg = FrictionConfig(
        spread_bps=3.0, impact_coeff=1.0,
        per_share_commission=0.005, min_commission=1.0,
        max_participation=0.05,
        borrow_rate_annual=datamod.fred_rate("SOFR"),  # live rate + loan spread; offline fallback
    )

    grid = {"fast": [10, 20, 40], "slow": [80, 100, 150],
            "vol_target": [0.10, 0.15]}

    wf = walk_forward(data, SYMBOL, grid, cfg,
                      is_days=756, oos_days=252, capital=1_000_000)

    stats = make_tearsheet(wf.oos_results, wf.trades, wf.costs,
                           path="tearsheet.png",
                           title="MA Cross (Vol-Targeted) — Stitched Out-of-Sample, Net of Frictions")

    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("=== Walk-forward folds (params chosen in-sample only) ===")
    print(wf.folds.to_string(index=False))
    print("\n=== Overfitting tax ===")
    print(f"Mean IS Sharpe : {wf.folds['is_sharpe'].mean():.2f}")
    print(f"Mean OOS Sharpe: {wf.folds['oos_sharpe'].mean():.2f}")
    dsr = deflated_sharpe(wf.oos_results["returns"], wf.n_trials, wf.trial_sharpes)
    print(f"Grid trials per fold: {wf.n_trials} | Expected max Sharpe under "
          f"zero skill (SR0): {dsr['SR0_ann']:.2f}")
    print(f"Deflated Sharpe Ratio: {dsr['DSR']:.1%}  "
          f"({'credible' if dsr['DSR'] > 0.95 else 'NOT distinguishable from luck'})")
    print("\n=== Stitched OOS summary (net of all frictions) ===")
    print(stats.to_string())
    wf.folds.to_csv("walk_forward_folds.csv", index=False)
    wf.trades.to_csv("trades.csv", index=False)

if __name__ == "__main__":
    main()
