"""Load and preprocess time series data from CSV files."""

import pandas as pd


def load_series(path: str, date_col: str = "date", value_col: str = "value") -> pd.Series:
    """
    Load a time series from a CSV file.

    Args:
        path: Path to the CSV file.
        date_col: Name of the column containing dates.
        value_col: Name of the column containing the values to forecast.

    Returns:
        A pandas Series indexed by date, sorted chronologically.
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col)
    series = df.set_index(date_col)[value_col]
    series.index.name = date_col
    return series


def train_test_split_series(series: pd.Series, test_size: int):
    """
    Split a time series into train/test sets, preserving order.

    Args:
        series: The full time series.
        test_size: Number of most recent observations to hold out for testing.

    Returns:
        Tuple of (train_series, test_series).
    """
    if test_size <= 0 or test_size >= len(series):
        raise ValueError("test_size must be between 1 and len(series) - 1")
    return series.iloc[:-test_size], series.iloc[-test_size:]
