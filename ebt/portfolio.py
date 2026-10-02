"""Portfolio: positions, cash, equity marking, borrow accrual, sizing.

Turns SignalEvents (target weights) into OrderEvents (share deltas),
applies FillEvents, and marks to market on every bar. Short positions
accrue borrow daily on the marked value of the short.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .data import DataHandler
from .events import FillEvent, OrderEvent, SignalEvent


class Portfolio:
    def __init__(self, data: DataHandler, initial_capital: float = 1_000_000,
                 borrow_rate_annual: float = 0.055, rebalance_band: float = 0.05):
        self.data = data
        self.initial_capital = initial_capital
        self.rebalance_band = rebalance_band  # skip trades < band * equity
        self.cash = initial_capital
        self.positions: dict[str, int] = defaultdict(int)
        self.borrow_rate_daily = borrow_rate_annual / 252
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        self.exposure_curve: list[tuple[pd.Timestamp, float]] = []
        self.trades: list[dict] = []
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_borrow = 0.0

    # ---- valuation --------------------------------------------------------
    def market_value(self) -> float:
        mv = 0.0
        for sym, qty in self.positions.items():
            bar = self.data.latest_bar(sym)
            if bar is not None:
                mv += qty * float(bar["close"])
        return mv

    @property
    def equity(self) -> float:
        return self.cash + self.market_value()

    # ---- event handlers -----------------------------------------------------
    def on_signal(self, sig: SignalEvent) -> OrderEvent | None:
        bar = self.data.latest_bar(sig.symbol)
        if bar is None:
            return None
        price = float(bar["close"])
        target_shares = int((sig.target_weight * self.equity) / price)
        delta = target_shares - self.positions[sig.symbol]
        # No-trade band: don't pay the spread to fine-tune tiny weight drift.
        if abs(delta) * price < self.rebalance_band * self.equity:
            return None
        return OrderEvent(timestamp=sig.timestamp, symbol=sig.symbol,
                          quantity=delta)

    def on_fill(self, fill: FillEvent) -> None:
        self.positions[fill.symbol] += fill.quantity
        self.cash -= fill.quantity * fill.fill_price
        self.cash -= fill.commission
        self.total_commission += fill.commission
        self.total_slippage += fill.slippage_cost
        self.trades.append({
            "timestamp": fill.timestamp, "symbol": fill.symbol,
            "quantity": fill.quantity, "price": fill.fill_price,
            "commission": fill.commission, "slippage": fill.slippage_cost,
        })

    def on_bar_close(self) -> None:
        """Mark to market and accrue borrow on shorts."""
        for sym, qty in self.positions.items():
            if qty < 0:
                bar = self.data.latest_bar(sym)
                if bar is not None:
                    charge = abs(qty) * float(bar["close"]) * self.borrow_rate_daily
                    self.cash -= charge
                    self.total_borrow += charge
        eq = self.equity
        gross = abs(self.market_value())
        self.equity_curve.append((self.data.now, eq))
        self.exposure_curve.append((self.data.now, gross / eq if eq > 0 else 0.0))

    # ---- outputs ------------------------------------------------------------
    def results(self) -> pd.DataFrame:
        eq = pd.Series(dict(self.equity_curve), name="equity").sort_index()
        exp = pd.Series(dict(self.exposure_curve), name="gross_exposure").sort_index()
        out = pd.concat([eq, exp], axis=1)
        out["returns"] = out["equity"].pct_change().fillna(0.0)
        return out

    def trade_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)
