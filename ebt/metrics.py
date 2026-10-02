"""Risk-adjusted performance metrics for the tearsheet."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan


def ann_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe(returns: pd.Series, rf_annual: float = 0.0) -> float:
    ex = returns - rf_annual / TRADING_DAYS
    return np.sqrt(TRADING_DAYS) * ex.mean() / ex.std() if ex.std() > 0 else np.nan


def sortino(returns: pd.Series, rf_annual: float = 0.0) -> float:
    ex = returns - rf_annual / TRADING_DAYS
    downside = ex[ex < 0].std()
    return np.sqrt(TRADING_DAYS) * ex.mean() / downside if downside and downside > 0 else np.nan


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    return drawdown_series(equity).min()


def rolling_sharpe(returns: pd.Series, window: int = 126) -> pd.Series:
    mu = returns.rolling(window).mean()
    sd = returns.rolling(window).std()
    return np.sqrt(TRADING_DAYS) * mu / sd


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def _norm_ppf(p: float) -> float:
    """Acklam's inverse-normal approximation (avoids a scipy dependency)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def deflated_sharpe(returns: pd.Series, n_trials: int,
                    trial_sharpes_ann: list[float]) -> dict[str, float]:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    Corrects the observed (per-period) Sharpe for the number of strategy
    configurations tried: under the null of zero skill, the *best* of
    n_trials will show a positive Sharpe by luck alone. SR0 is that
    expected-maximum benchmark; DSR is the probability the observed Sharpe
    genuinely exceeds it, adjusting for return skew and kurtosis.
    DSR > 0.95 is the conventional bar for taking a backtest seriously.
    """
    r = returns.dropna()
    T = len(r)
    if T < 30 or r.std() == 0:
        return {"SR0_ann": np.nan, "DSR": np.nan}
    sr = r.mean() / r.std()                      # per-period
    trials = np.array(trial_sharpes_ann, dtype=float) / np.sqrt(TRADING_DAYS)
    trials = trials[~np.isnan(trials)]
    sr_var = trials.var(ddof=1) if len(trials) > 1 else (sr ** 2 or 1e-8)
    emc = 0.5772156649
    n = max(n_trials, 2)
    sr0 = np.sqrt(sr_var) * ((1 - emc) * _norm_ppf(1 - 1 / n)
                             + emc * _norm_ppf(1 - 1 / (n * np.e)))
    skew = float(r.skew())
    kurt = float(r.kurtosis()) + 3  # pandas gives excess kurtosis
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    dsr = _norm_cdf((sr - sr0) * np.sqrt(T - 1) / denom)
    return {"SR0_ann": sr0 * np.sqrt(TRADING_DAYS), "DSR": dsr}


def summary(results: pd.DataFrame, trades: pd.DataFrame,
            costs: dict[str, float]) -> pd.Series:
    eq, rets = results["equity"], results["returns"]
    n_trades = len(trades)
    turnover = (trades["quantity"].abs() * trades["price"]).sum() / eq.mean() \
        if n_trades else 0.0
    return pd.Series({
        "CAGR": cagr(eq),
        "Ann. Vol": ann_vol(rets),
        "Sharpe": sharpe(rets),
        "Sortino": sortino(rets),
        "Max Drawdown": max_drawdown(eq),
        "Calmar": cagr(eq) / abs(max_drawdown(eq)) if max_drawdown(eq) != 0 else np.nan,
        "Trades": n_trades,
        "Turnover (x avg equity)": turnover,
        "Commissions ($)": costs.get("commission", 0.0),
        "Slippage ($)": costs.get("slippage", 0.0),
        "Borrow ($)": costs.get("borrow", 0.0),
        "Final Equity ($)": eq.iloc[-1],
    })
