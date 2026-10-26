"""Simple baseline forecasting models."""

import pandas as pd


def naive_forecast(series: pd.Series, horizon: int) -> pd.Series:
    """
    Naive forecast: repeats the last observed value for `horizon` steps.
    """
    last_value = series.iloc[-1]
    future_index = _future_index(series, horizon)
    return pd.Series([last_value] * horizon, index=future_index, name="forecast")


def moving_average_forecast(series: pd.Series, window: int, horizon: int) -> pd.Series:
    """
    Moving average forecast: repeats the average of the last `window`
    observations for `horizon` steps.
    """
    if window <= 0 or window > len(series):
        raise ValueError("window must be between 1 and len(series)")
    avg = series.iloc[-window:].mean()
    future_index = _future_index(series, horizon)
    return pd.Series([avg] * horizon, index=future_index, name="forecast")


def _future_index(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """Build a future date index continuing from the end of `series`."""
    freq = pd.infer_freq(series.index) or "D"
    start = series.index[-1]
    return pd.date_range(start=start, periods=horizon + 1, freq=freq)[1:]
