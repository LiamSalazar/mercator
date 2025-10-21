from __future__ import annotations
import numpy as np
import pandas as pd

def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Volatilidad de rango Parkinson (1980). Mejor que close-to-close si hay gaps."""
    rng = (high / low).apply(np.log)  # ln(H/L)
    coef = 1.0 / (4.0 * np.log(2.0))
    return (coef * (rng ** 2)).rolling(window).sum().pow(0.5)

def garman_klass_vol(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Volatilidad Garman-Klass (1980): incorpora open y close (más eficiente)."""
    log_hl = (high / low).apply(np.log)
    log_co = (close / open_).apply(np.log)
    var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    var = var.rolling(window).sum()
    return var.clip(lower=0).pow(0.5)

def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()

def add_range_vol_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Espera columnas: ['date','Ticker','Open','High','Low','Close','Volume']
    Devuelve DataFrame con retornos y volatilidades (PK, GK).
    """
    def _by_ticker(x: pd.DataFrame) -> pd.DataFrame:
        x = x.sort_values("date").copy()
        x["ret_log_1d"] = log_returns(x["Close"])
        x["vol_pk"] = parkinson_vol(x["High"], x["Low"], window=window)
        x["vol_gk"] = garman_klass_vol(x["Open"], x["High"], x["Low"], x["Close"], window=window)
        return x

    out = df.groupby("Ticker", group_keys=False).apply(_by_ticker)
    return out
