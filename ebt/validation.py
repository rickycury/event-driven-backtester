"""Walk-forward validation.

Anchored-window protocol:
  for each fold: fit params on the in-sample (IS) window by grid search,
  then run those params — frozen — on the following out-of-sample (OOS)
  window. The stitched OOS equity curve is the only number you're allowed
  to believe. If IS Sharpe is 2.0 and stitched OOS is 0.4, the difference
  is your overfitting tax, quantified.

Selection metric is IS Sharpe; nothing from any OOS window ever touches
parameter choice.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from .data import DataHandler
from .engine import BacktestEngine
from .execution import FrictionConfig, SimulatedExecutionHandler
from .metrics import sharpe
from .portfolio import Portfolio
from .strategy import MACrossVolTarget


def _run_window(df: pd.DataFrame, symbol: str, params: dict,
                cfg: FrictionConfig, capital: float) -> pd.DataFrame:
    data = DataHandler({symbol: df})
    strat = MACrossVolTarget(**params)
    port = Portfolio(data, initial_capital=capital,
                     borrow_rate_annual=cfg.borrow_rate_annual)
    execu = SimulatedExecutionHandler(data, cfg)
    BacktestEngine(data, strat, port, execu).run()
    return port.results(), port


@dataclass
class WalkForwardResult:
    oos_results: pd.DataFrame        # stitched OOS equity/returns/exposure
    folds: pd.DataFrame              # per-fold chosen params + IS/OOS Sharpe
    trades: pd.DataFrame
    costs: dict
    n_trials: int                    # grid configurations evaluated per fold
    trial_sharpes: list[float]       # every IS Sharpe from every fold's grid


def walk_forward(data: DataHandler, symbol: str, param_grid: dict,
                 cfg: FrictionConfig, is_days: int = 756, oos_days: int = 252,
                 capital: float = 1_000_000) -> WalkForwardResult:
    df = data.frames[symbol]
    keys = list(param_grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]

    fold_rows, oos_chunks, all_trades = [], [], []
    trial_sharpes: list[float] = []
    costs = {"commission": 0.0, "slippage": 0.0, "borrow": 0.0}
    start, fold = 0, 0
    rolling_capital = capital

    while start + is_days + oos_days <= len(df):
        is_df = df.iloc[start: start + is_days]
        oos_df = df.iloc[start + is_days - max(p for p in param_grid.get("slow", [100])):
                         start + is_days + oos_days]  # warm-up prefix, trades gated below

        # --- fit: grid search on IS only ---
        best_params, best_sh = None, -1e9
        for params in combos:
            if params.get("fast", 0) >= params.get("slow", 1):
                continue
            res, _ = _run_window(is_df, symbol, params, cfg, capital)
            sh = sharpe(res["returns"])
            trial_sharpes.append(sh)
            if pd.notna(sh) and sh > best_sh:
                best_sh, best_params = sh, params

        # --- test: frozen params on OOS (warm-up bars excluded from P&L) ---
        oos_start_ts = df.index[start + is_days]
        res, port = _run_window(oos_df, symbol, best_params, cfg, rolling_capital)
        res = res.loc[res.index >= oos_start_ts]
        oos_sh = sharpe(res["returns"])

        oos_chunks.append(res)
        tf = port.trade_frame()
        if not tf.empty:
            all_trades.append(tf[tf["timestamp"] >= oos_start_ts])
        costs["commission"] += port.total_commission
        costs["slippage"] += port.total_slippage
        costs["borrow"] += port.total_borrow
        if not res.empty:
            rolling_capital = float(res["equity"].iloc[-1])

        fold_rows.append({"fold": fold,
                          "is_start": is_df.index[0].date(),
                          "oos_start": oos_start_ts.date(),
                          "oos_end": res.index[-1].date() if not res.empty else None,
                          **{f"param_{k}": v for k, v in best_params.items()},
                          "is_sharpe": best_sh, "oos_sharpe": oos_sh})
        start += oos_days
        fold += 1

    oos = pd.concat(oos_chunks)
    # re-derive stitched returns so chunk boundaries don't inject zeros
    oos["returns"] = oos["equity"].pct_change().fillna(0.0)
    trades = pd.concat(all_trades) if all_trades else pd.DataFrame(
        columns=["timestamp", "symbol", "quantity", "price", "commission", "slippage"])
    return WalkForwardResult(oos_results=oos, folds=pd.DataFrame(fold_rows),
                             trades=trades, costs=costs,
                             n_trials=len(combos), trial_sharpes=trial_sharpes)
