import time
import os

from fastapi import FastAPI, HTTPException, Query
from starlette.responses import JSONResponse
from fastapi import Request
from fastapi_cache.decorator import cache
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
import redis

try:
    from .model import (
        add_indicators,
        compute_stats,
        download_stock,
        evaluate_model,
        prepare_training_data,
        train_model,
    )
except ImportError:
    from model import (
        add_indicators,
        compute_stats,
        download_stock,
        evaluate_model,
        prepare_training_data,
        train_model,
    )

app = FastAPI()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

@app.on_event("startup")
async def init_cache():
    FastAPICache.init(RedisBackend(redis_client), prefix="stock-api",cache_status_header="X-Cache",)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


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
        if len(data) < 200:
            raise ValueError("Not enough historical data to compute reliable indicators.")

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
@cache(expire=3600)  # cache for 1 hour
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
            "debug_ts": time.time(),  # <-- added

        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
