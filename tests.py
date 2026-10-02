"""Sanity tests: run with `python tests.py`.

1. Causality: history() never leaks bars past the clock.
2. Capacity: an oversized order is clipped at 5% participation and the
   residual keeps filling on later opens.
3. Signed FIFO: short round-trips book PnL with the correct sign.
4. DSR: more trials => higher luck benchmark => lower DSR.
"""
import numpy as np
import pandas as pd

from ebt.data import DataHandler, synthetic
from ebt.execution import FrictionConfig, SimulatedExecutionHandler
from ebt.events import OrderEvent
from ebt.metrics import deflated_sharpe
from ebt.tearsheet import trade_pnls


def test_causality():
    data = synthetic(n_days=50, seed=1)
    sym = data.symbols[0]
    for _ in range(30):
        data.step()
        assert data.history(sym).index.max() <= data.now
    print("ok  causality: history() never exceeds the clock")


def test_capacity_and_working():
    idx = pd.bdate_range("2024-01-01", periods=10)
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                       "close": 100.0, "volume": 10_000}, index=idx)
    data = DataHandler({"X": df})
    ex = SimulatedExecutionHandler(data, FrictionConfig(max_participation=0.05))
    data.step()
    ex.submit(OrderEvent(timestamp=data.now, symbol="X", quantity=1200))
    filled = []
    for _ in range(4):
        data.step()
        filled += [f.quantity for f in ex.on_bar_open()]
    assert filled == [500, 500, 200], filled  # 5% of 10k, worked over 3 opens
    print("ok  capacity: 1200 shares filled as", filled)


def test_signed_fifo():
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2),
        "quantity": [-100, 100], "price": [50.0, 40.0]})
    pnl = trade_pnls(trades)
    assert np.isclose(pnl.iloc[0], 1000.0), pnl  # short 50, cover 40 => +1000
    print("ok  fifo: short round-trip books +$1,000")


def test_dsr_monotone():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0004, 0.01, 1000))
    trials = list(rng.normal(0.3, 0.4, 50))
    d_small = deflated_sharpe(rets, 2, trials)["DSR"]
    d_big = deflated_sharpe(rets, 500, trials)["DSR"]
    assert d_big < d_small, (d_small, d_big)
    print(f"ok  dsr: 2 trials -> {d_small:.2%}, 500 trials -> {d_big:.2%}")


if __name__ == "__main__":
    test_causality()
    test_capacity_and_working()
    test_signed_fifo()
    test_dsr_monotone()
    print("all tests passed")
