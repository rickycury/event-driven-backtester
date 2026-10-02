"""Strategies see only DataHandler.history() — the causal window.

Sample: dual moving-average trend follower with a volatility-targeted
weight. Deliberately simple; the point of the project is the plumbing,
not the alpha.
"""
from __future__ import annotations

import numpy as np

from .data import DataHandler
from .events import MarketEvent, SignalEvent


class Strategy:
    def on_market(self, event: MarketEvent, data: DataHandler) -> SignalEvent | None:
        raise NotImplementedError


class MACrossVolTarget(Strategy):
    def __init__(self, fast: int = 20, slow: int = 100,
                 vol_target: float = 0.15, max_weight: float = 1.0):
        assert fast < slow
        self.fast, self.slow = fast, slow
        self.vol_target, self.max_weight = vol_target, max_weight

    def on_market(self, event: MarketEvent, data: DataHandler) -> SignalEvent | None:
        hist = data.history(event.symbol, self.slow + 1)
        if len(hist) < self.slow + 1:
            return None
        close = hist["close"]
        fast_ma = close.tail(self.fast).mean()
        slow_ma = close.tail(self.slow).mean()
        direction = 1.0 if fast_ma > slow_ma else 0.0  # long/flat

        rets = close.pct_change().dropna().tail(63)
        realized = rets.std() * np.sqrt(252)
        scale = min(self.vol_target / realized, self.max_weight) if realized > 0 else 0.0

        return SignalEvent(timestamp=event.timestamp, symbol=event.symbol,
                           target_weight=direction * scale)
