import math
from typing import Optional

import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA50",
    "Daily Return",
    "Volatility50",
    "Volatility200",
]


def _to_series(column: pd.Series) -> pd.Series:
    if hasattr(column, "columns"):
        return column.iloc[:, 0]
    return column


def _validate_time_index(df: pd.DataFrame) -> None:
    if df.index.has_duplicates:
        raise ValueError("Stock data index must not contain duplicate dates.")

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Stock data index must be sorted in chronological order."
        )


def download_stock(
    ticker: str,
    start: str = "2023-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    df = yf.download(
        ticker.upper(),
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        raise ValueError(f"No stock data returned for ticker '{ticker}'.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df = df[required_columns].copy()

    for column in required_columns:
        df[column] = pd.to_numeric(
            _to_series(df[column]),
            errors="coerce",
        )

    df = df.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    df = df.dropna(subset=required_columns)

    if df.empty:
        raise ValueError("No valid stock rows remain after cleaning.")

    _validate_time_index(df)

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    _validate_time_index(df)

    data = df.copy()

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        data[column] = _to_series(data[column])

    close = data["Close"]

    data["MA50"] = close.rolling(50).mean()

    data["Daily Return"] = close.pct_change()

    data["Volatility50"] = (
        data["Daily Return"]
        .rolling(50)
        .std()
    )

    data["Volatility200"] = (
        data["Daily Return"]
        .rolling(200)
        .std()
    )

    data = data.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    return data


def compute_stats(df: pd.DataFrame) -> dict:
    _validate_time_index(df)

    close = _to_series(df["Close"]).astype(float)
    high = _to_series(df["High"]).astype(float)
    low = _to_series(df["Low"]).astype(float)

    if len(close) < 2:
        raise ValueError(
            "At least two price rows are required to compute returns."
        )

    if close.iloc[0] == 0:
        raise ValueError(
            "The first closing price cannot be zero when computing return."
        )

    total_return = (
        close.iloc[-1] - close.iloc[0]
    ) / close.iloc[0]

    trading_days = len(close)

    annual_return = (
        (1 + total_return) ** (252 / trading_days)
    ) - 1

    result = {
        "highest_price": high.max(),
        "lowest_price": low.min(),
        "recent_price": close.iloc[-1],
        "total_return": total_return,
        "annualized_return": annual_return,
    }

    for name, value in result.items():
        if not math.isfinite(float(value)):
            raise ValueError(
                f"Calculated statistic '{name}' is invalid."
            )

    return {
        name: float(value)
        for name, value in result.items()
    }


def prepare_training_data(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
):
    del random_state

    _validate_time_index(df)

    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    data = df.copy()

    for column in ["Close", "Open", "High", "Low", "Volume"]:
        data[column] = _to_series(data[column])

    feature_frame = data[FEATURE_COLUMNS].dropna().copy()

    if feature_frame.empty:
        raise ValueError(
            "Not enough clean rows after indicator generation."
        )

    latest_features = feature_frame.iloc[-1].copy()

    feature_frame["target"] = data["Close"].shift(-1)

    supervised = feature_frame.dropna().copy()

    if len(supervised) < 20:
        raise ValueError(
            "Not enough rows to train model. "
            "Try a ticker with longer history."
        )

    test_rows = math.ceil(len(supervised) * test_size)
    train_rows = len(supervised) - test_rows

    if test_rows < 2:
        raise ValueError(
            "test_size creates fewer than two test rows; "
            "R2 requires at least two test observations."
        )

    if train_rows < 1:
        raise ValueError(
            "test_size leaves no rows for model training."
        )

    X = supervised[FEATURE_COLUMNS]
    y = supervised["target"]

    X_train = X.iloc[:train_rows]
    X_test = X.iloc[train_rows:]

    y_train = y.iloc[:train_rows]
    y_test = y.iloc[train_rows:]

    return X_train, X_test, y_train, y_test, latest_features


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    if len(y_test) < 2:
        raise ValueError(
            "At least two test rows are required to calculate R2."
        )

    predictions = model.predict(X_test)

    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }