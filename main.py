"""Example entry point: load data, forecast, and evaluate."""

import argparse
import sys

sys.path.insert(0, "src")

from time_series_forecasting.data_loader import load_series, train_test_split_series
from time_series_forecasting.models import moving_average_forecast
from time_series_forecasting.evaluate import evaluate_forecast


def main():
    parser = argparse.ArgumentParser(description="Simple time series forecasting demo")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--date-col", default="date", help="Name of the date column")
    parser.add_argument("--value-col", default="value", help="Name of the value column")
    parser.add_argument("--horizon", type=int, default=7, help="Number of periods to forecast")
    parser.add_argument("--window", type=int, default=3, help="Moving average window size")
    args = parser.parse_args()

    series = load_series(args.input, date_col=args.date_col, value_col=args.value_col)

    if len(series) > args.horizon:
        train, test = train_test_split_series(series, test_size=args.horizon)
        forecast = moving_average_forecast(train, window=args.window, horizon=args.horizon)
        metrics = evaluate_forecast(test.values, forecast.values)
        print("Holdout evaluation:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        print()

    full_forecast = moving_average_forecast(series, window=args.window, horizon=args.horizon)
    print(f"Forecast for next {args.horizon} periods:")
    print(full_forecast)


if __name__ == "__main__":
    main()
