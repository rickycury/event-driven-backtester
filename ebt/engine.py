"""The event loop.

Per bar t:
  1. Clock advances; MarketEvents for symbols with a bar at t.
  2. Strategy consumes MarketEvents  -> SignalEvents   (info as of t only)
  3. Portfolio sizes SignalEvents    -> OrderEvents
  4. Execution fills OrderEvents at bar t+1's OPEN     (the causality gap)
  5. Portfolio applies fills, then marks to market and accrues borrow.

Note the fills are applied with t+1's price but the decision used only
t's information — signal on close, trade next open. That one design
choice removes the single most common lookahead bug in retail code.
"""
from __future__ import annotations

from collections import deque

from .data import DataHandler
from .events import EventType, MarketEvent
from .execution import SimulatedExecutionHandler
from .portfolio import Portfolio
from .strategy import Strategy


class BacktestEngine:
    def __init__(self, data: DataHandler, strategy: Strategy,
                 portfolio: Portfolio, execution: SimulatedExecutionHandler):
        self.data, self.strategy = data, strategy
        self.portfolio, self.execution = portfolio, execution

    def run(self) -> None:
        while True:
            ts = self.data.step()
            if ts is None:
                break

            # 1. Bar opens: resting orders (decided on prior closes) fill
            #    at this bar's open, capacity permitting; residuals rest on.
            for fill in self.execution.on_bar_open():
                self.portfolio.on_fill(fill)

            # 2. Bar closes: strategy sees info through ts, new orders are
            #    submitted to rest until the NEXT open — the causality gap.
            queue: deque = deque()
            for sym in self.data.symbols:
                if ts in self.data.frames[sym].index:
                    queue.append(MarketEvent(timestamp=ts, symbol=sym))

            while queue:
                ev = queue.popleft()
                if ev.type is EventType.MARKET:
                    sig = self.strategy.on_market(ev, self.data)
                    if sig:
                        queue.append(sig)
                elif ev.type is EventType.SIGNAL:
                    order = self.portfolio.on_signal(ev)
                    if order:
                        self.execution.submit(order)

            self.portfolio.on_bar_close()
