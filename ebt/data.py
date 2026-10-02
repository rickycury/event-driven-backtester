"""Data layer.

The DataHandler owns the full price history but only ever *exposes*
bars up to and including the current simulation clock. Strategies are
handed the handler, not the raw DataFrame, so peeking into the future
is an API violation rather than an easy mistake.

Loaders provided:
  - from_dataframe / from_csv  : bring your own OHLCV
  - from_yahoo                 : thin yfinance wrapper (network required)
  - synthetic                  : regime-switching GBM with volume, for
                                 fully reproducible offline runs
"""
from __future__ import annotations

import numpy as np
import pandas as pd

def fred_rate(series_id: str = "SOFR", fallback: float = 0.043,
              stock_loan_spread: float = 0.010, timeout: int = 10) -> float:
    """Latest financing rate from FRED (no API key needed) plus a
    stock-loan spread, as an annualized decimal for borrow accrual.

    Uses the public fredgraph.csv endpoint; falls back to `fallback` +
    spread when offline so backtests remain reproducible without network.
    Sensible series: SOFR, DFF (fed funds), DGS3MO.
    """
    import io
    import urllib.request
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            df = pd.read_csv(io.StringIO(resp.read().decode()))
        rate = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().iloc[-1] / 100
        print(f"[fred] {series_id} = {rate:.2%} (+{stock_loan_spread:.2%} loan spread)")
        return float(rate) + stock_loan_spread
    except Exception as e:
        print(f"[fred] fetch failed ({type(e).__name__}); "
              f"using fallback {fallback:.2%} + {stock_loan_spread:.2%}")
        return fallback + stock_loan_spread


REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


class DataHandler:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        for sym, df in frames.items():
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                raise ValueError(f"{sym} missing columns: {missing}")
            if not df.index.is_monotonic_increasing:
                raise ValueError(f"{sym} index must be sorted ascending")
        self.frames = {s: df[REQUIRED_COLS].copy() for s, df in frames.items()}
        self.symbols = list(self.frames)
        # Union calendar drives the event loop.
        self.calendar = sorted(set().union(*[df.index for df in self.frames.values()]))
        self._cursor = -1  # index into calendar; -1 == before first bar

    # ---- clock ------------------------------------------------------------
    def step(self) -> pd.Timestamp | None:
        """Advance the clock one bar. Returns new timestamp or None at end."""
        if self._cursor + 1 >= len(self.calendar):
            return None
        self._cursor += 1
        return self.calendar[self._cursor]

    @property
    def now(self) -> pd.Timestamp:
        if self._cursor < 0:
            raise RuntimeError("Clock not started; call step() first.")
        return self.calendar[self._cursor]

    # ---- causal accessors ---------------------------------------------------
    def history(self, symbol: str, n: int | None = None) -> pd.DataFrame:
        """Bars up to and including `now`. Never anything later."""
        df = self.frames[symbol]
        visible = df.loc[: self.now]
        return visible if n is None else visible.tail(n)

    def latest_bar(self, symbol: str) -> pd.Series | None:
        h = self.history(symbol, 1)
        return None if h.empty else h.iloc[-1]

    def next_bar(self, symbol: str) -> pd.Series | None:
        """The bar AFTER `now` — reserved for the execution handler only.

        Orders decided on bar t fill at bar t+1's open; the broker is the
        one entity legitimately allowed to see the next bar's prices.
        """
        df = self.frames[symbol]
        future = df.loc[df.index > self.now]
        return None if future.empty else future.iloc[0]

    def adv(self, symbol: str, window: int = 21) -> float:
        """Trailing average daily dollar volume (capacity anchor)."""
        h = self.history(symbol, window)
        if h.empty:
            return 0.0
        return float((h["close"] * h["volume"]).mean())


# ---- loaders ----------------------------------------------------------------
def from_csv(path: str, symbol: str) -> DataHandler:
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.columns = [c.lower() for c in df.columns]
    return DataHandler({symbol: df})


def from_yahoo(symbols: list[str], start: str, end: str) -> DataHandler:
    import yfinance as yf  # optional dependency
    frames = {}
    for s in symbols:
        raw = yf.download(s, start=start, end=end, auto_adjust=True, progress=False)
        raw.columns = [c.lower() for c in raw.columns]
        frames[s] = raw
    return DataHandler(frames)


def synthetic(symbol: str = "SYN", n_days: int = 2520, seed: int = 7,
              s0: float = 100.0) -> DataHandler:
    """Regime-switching GBM daily bars with plausible volume.

    Two regimes (calm uptrend / volatile drawdown) via a 2-state Markov
    chain, so momentum has something real to chew on out of sample.
    """
    rng = np.random.default_rng(seed)
    # regimes: 0 = bull (mu 12%, vol 14%), 1 = bear (mu -18%, vol 32%)
    mu = np.array([0.12, -0.18]) / 252
    sig = np.array([0.14, 0.32]) / np.sqrt(252)
    P = np.array([[0.995, 0.005], [0.015, 0.985]])
    state = 0
    rets = np.empty(n_days)
    for i in range(n_days):
        rets[i] = rng.normal(mu[state], sig[state])
        state = rng.choice(2, p=P[state])
    close = s0 * np.exp(np.cumsum(rets))
    o = close * np.exp(rng.normal(0, 0.002, n_days))
    intraday = np.abs(rng.normal(0, 0.008, n_days))
    hi = np.maximum(o, close) * (1 + intraday)
    lo = np.minimum(o, close) * (1 - intraday)
    vol = (rng.lognormal(mean=13.5, sigma=0.35, size=n_days)
           * (1 + 3 * np.abs(rets) / rets.std())).astype(int)
    idx = pd.bdate_range("2016-01-04", periods=n_days)
    df = pd.DataFrame({"open": o, "high": hi, "low": lo,
                       "close": close, "volume": vol}, index=idx)
    return DataHandler({symbol: df})
