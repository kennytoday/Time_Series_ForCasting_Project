import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from time_series_forecasting.models import naive_forecast, moving_average_forecast
from time_series_forecasting.evaluate import evaluate_forecast


def _sample_series():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    values = [10, 12, 11, 13, 15, 14, 16, 18, 17, 19]
    return pd.Series(values, index=dates)


def test_naive_forecast_length_and_value():
    series = _sample_series()
    forecast = naive_forecast(series, horizon=3)
    assert len(forecast) == 3
    assert (forecast == series.iloc[-1]).all()


def test_moving_average_forecast_length_and_value():
    series = _sample_series()
    forecast = moving_average_forecast(series, window=3, horizon=4)
    expected_avg = series.iloc[-3:].mean()
    assert len(forecast) == 4
    assert all(abs(v - expected_avg) < 1e-9 for v in forecast)


def test_evaluate_forecast_perfect_prediction():
    actual = pd.Series([1, 2, 3])
    predicted = pd.Series([1, 2, 3])
    metrics = evaluate_forecast(actual, predicted)
    assert metrics["MAE"] == 0
    assert metrics["RMSE"] == 0
    assert metrics["MAPE"] == 0
