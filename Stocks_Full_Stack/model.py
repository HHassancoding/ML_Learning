import pandas as pd
import yfinance as yf
from typing import Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


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


def download_stock(ticker: str, start: str = "2023-01-01", end: Optional[str] = None) -> pd.DataFrame:
    df = yf.download(ticker.upper(), start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No stock data returned for ticker '{ticker}'.")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Open"] = _to_series(data["Open"])
    data["High"] = _to_series(data["High"])
    data["Low"] = _to_series(data["Low"])
    data["Close"] = _to_series(data["Close"])
    data["Volume"] = _to_series(data["Volume"])
    close = data["Close"]
    data["MA50"] = close.rolling(50).mean()
    data["Daily Return"] = close.pct_change()
    data["Volatility50"] = data["Daily Return"].rolling(50).std()
    data["Volatility200"] = data["Daily Return"].rolling(200).std()
    return data


def compute_stats(df: pd.DataFrame) -> dict:
    close = _to_series(df["Close"])
    high = _to_series(df["High"])
    low = _to_series(df["Low"])

    total_return = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
    trading_days = len(close)
    annual_return = (1 + total_return) ** (252 / trading_days) - 1

    return {
        "highest_price": float(high.max()),
        "lowest_price": float(low.min()),
        "recent_price": float(close.iloc[-1]),
        "total_return": float(total_return),
        "annualized_return": float(annual_return),
    }


def prepare_training_data(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
):
    data = df.copy()
    data["Close"] = _to_series(data["Close"])
    data["Open"] = _to_series(data["Open"])
    data["High"] = _to_series(data["High"])
    data["Low"] = _to_series(data["Low"])
    data["Volume"] = _to_series(data["Volume"])

    feature_frame = data[FEATURE_COLUMNS].dropna().copy()
    if feature_frame.empty:
        raise ValueError("Not enough clean rows after indicator generation.")

    latest_features = feature_frame.iloc[-1]
    feature_frame["target"] = data["Close"].shift(-1)
    supervised = feature_frame.dropna().copy()

    if len(supervised) < 20:
        raise ValueError("Not enough rows to train model. Try a ticker with longer history.")

    X = supervised[FEATURE_COLUMNS]
    y = supervised["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test, latest_features


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }
