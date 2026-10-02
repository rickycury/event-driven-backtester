"""Execution layer: the part most retail backtests skip.

Fill model (next-bar-open):
  reference price  = next bar's open (order decided at t, filled at t+1)
  half-spread cost = spread_bps / 2, always paid
  market impact    = impact_coeff * sqrt(participation), Almgren-style
                     square-root law, where participation = |shares| / bar volume
  commission       = max(min_commission, per_share * |shares|)
  capacity         = fills are clipped to max_participation of the bar's
                     volume; the remainder is simply not done (no magic
                     liquidity), which is how capacity constraints bite.

Borrow: short positions accrue a financing charge daily inside the
Portfolio, at borrow_rate_annual (feed FRED's rate + a spread here).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .data import DataHandler
from .events import FillEvent, OrderEvent


@dataclass
class FrictionConfig:
    spread_bps: float = 3.0            # full quoted spread, bps of price
    impact_coeff: float = 1.0          # Almgren eta: impact = eta * sigma_d * sqrt(part.)
    per_share_commission: float = 0.005
    min_commission: float = 1.0
    max_participation: float = 0.05    # ≤5% of the bar's volume
    borrow_rate_annual: float = 0.055  # fed funds proxy + stock-loan spread


class SimulatedExecutionHandler:
    """Holds a resting order book. Orders submitted after bar t's close
    begin filling at bar t+1's open; anything capacity denies rests and
    keeps working at subsequent opens. New orders for a symbol cancel and
    replace the resting one (the portfolio re-derives the delta from its
    true position each signal, so replacement is the correct semantics).
    """

    def __init__(self, data: DataHandler, cfg: FrictionConfig):
        self.data = data
        self.cfg = cfg
        self._working: dict[str, int] = {}   # symbol -> signed residual shares

    def submit(self, order: OrderEvent) -> None:
        if order.quantity != 0:
            self._working[order.symbol] = order.quantity  # cancel-replace

    def on_bar_open(self) -> list[FillEvent]:
        """Fill resting orders at the CURRENT bar's open, capacity-clipped."""
        fills = []
        for sym in list(self._working):
            bar = self.data.latest_bar(sym)
            if bar is None:
                continue
            remaining = self._working[sym]
            cap = int(self.cfg.max_participation * bar["volume"])
            qty = int(math.copysign(min(abs(remaining), cap), remaining))
            if qty == 0:
                continue

            ref = float(bar["open"])
            participation = abs(qty) / max(bar["volume"], 1)

            # Almgren square-root law: impact scales with the stock's own
            # daily volatility — trading 5% of ADV in a sleepy utility is
            # not the same as 5% of ADV in a biotech. Vol estimated causally.
            hist = self.data.history(sym, 22)["close"]
            sigma_d = float(hist.pct_change().dropna().std()) if len(hist) > 5 else 0.02

            half_spread = ref * (self.cfg.spread_bps / 2) / 1e4
            impact = ref * self.cfg.impact_coeff * sigma_d * math.sqrt(participation)
            adverse = half_spread + impact
            fill_price = ref + adverse if qty > 0 else ref - adverse

            commission = max(self.cfg.min_commission,
                             self.cfg.per_share_commission * abs(qty))
            fills.append(FillEvent(
                timestamp=self.data.now, symbol=sym, quantity=qty,
                fill_price=fill_price, commission=commission,
                slippage_cost=adverse * abs(qty)))

            residual = remaining - qty
            if residual == 0:
                del self._working[sym]
            else:
                self._working[sym] = residual
        return fills
