# Time Series Forecasting

A simple Python project for forecasting time series data (e.g. sales, prices, demand)
using classical and lightweight ML models.

## Features

- Load and clean time series data from CSV
- Basic visualization (trend, seasonality)
- Train/test split respecting time order
- Baseline forecasting models (naive, moving average)
- Simple model evaluation (MAE, RMSE, MAPE)

## Project Structure

```
time_series_forecasting/
├── data/                      # Raw and processed data (CSV files)
├── notebooks/                 # Exploratory analysis notebooks
├── src/
│   └── time_series_forecasting/
│       ├── __init__.py
│       ├── data_loader.py     # Load and preprocess data
│       ├── models.py          # Forecasting models
│       └── evaluate.py        # Evaluation metrics
├── tests/
│   └── test_models.py         # Unit tests
├── requirements.txt
├── main.py                    # Example entry point
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Place your time series CSV in `data/` with at least a date column and a value column,
then run:

```bash
python main.py --input data/sample.csv --date-col date --value-col value --horizon 7
```

This will print forecasted values for the next `horizon` periods along with basic
evaluation metrics on a holdout set.

## Example

```python
from time_series_forecasting.data_loader import load_series
from time_series_forecasting.models import moving_average_forecast
from time_series_forecasting.evaluate import evaluate_forecast

series = load_series("data/sample.csv", date_col="date", value_col="value")
forecast = moving_average_forecast(series, window=3, horizon=7)
print(forecast)
```

## Running Tests

```bash
pytest tests/
```

## License

MIT

python main.py --input data/sample.csv --horizon 5
