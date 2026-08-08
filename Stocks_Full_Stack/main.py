from fastapi import FastAPI, HTTPException, Query
from model import (
    add_indicators,
    compute_stats,
    download_stock,
    evaluate_model,
    prepare_training_data,
    train_model,
)

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Stock Prediction API is running"}


def _safe_float(value):
    if value != value:  # NaN check
        return None
    return float(value)


@app.get("/stats")
def stats(ticker: str = Query(..., min_length=1)):
    try:
        data = download_stock(ticker)
        payload = compute_stats(data)
        return {"ticker": ticker.upper(), **payload}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/indicators")
def indicators(ticker: str = Query(..., min_length=1)):
    try:
        data = add_indicators(download_stock(ticker))
        cleaned = data.dropna(subset=["MA50", "Daily Return", "Volatility50", "Volatility200"])
        if cleaned.empty:
            raise ValueError("Not enough data to compute indicators.")

        latest = cleaned.iloc[-1]
        recent = cleaned.tail(10)

        return {
            "ticker": ticker.upper(),
            "latest": {
                "ma50": _safe_float(latest["MA50"]),
                "daily_return": _safe_float(latest["Daily Return"]),
                "volatility50": _safe_float(latest["Volatility50"]),
                "volatility200": _safe_float(latest["Volatility200"]),
            },
            "recent": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "close": _safe_float(row["Close"]),
                    "ma50": _safe_float(row["MA50"]),
                    "daily_return": _safe_float(row["Daily Return"]),
                    "volatility50": _safe_float(row["Volatility50"]),
                    "volatility200": _safe_float(row["Volatility200"]),
                }
                for idx, row in recent.iterrows()
            ],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/predict")
def predict(ticker: str = Query(..., min_length=1)):
    try:
        data = add_indicators(download_stock(ticker))
        X_train, X_test, y_train, y_test, latest_features = prepare_training_data(data)
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        next_day_prediction = model.predict([latest_features])[0]

        return {
            "ticker": ticker.upper(),
            "predicted_next_day_close": float(next_day_prediction),
            "metrics": metrics,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
