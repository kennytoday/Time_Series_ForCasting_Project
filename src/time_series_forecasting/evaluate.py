"""Basic evaluation metrics for forecasts."""

import numpy as np
import pandas as pd


def evaluate_forecast(actual: pd.Series, predicted: pd.Series) -> dict:
    """
    Compute basic error metrics between actual and predicted values.

    Args:
        actual: Ground-truth values.
        predicted: Forecasted values, same length/order as `actual`.

    Returns:
        Dict with MAE, RMSE, and MAPE.
    """
    actual_vals = np.asarray(actual, dtype=float)
    predicted_vals = np.asarray(predicted, dtype=float)

    errors = actual_vals - predicted_vals
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    nonzero = actual_vals != 0
    mape = (
        np.mean(np.abs(errors[nonzero] / actual_vals[nonzero])) * 100
        if nonzero.any()
        else float("nan")
    )

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
