from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, parse, request

import pandas as pd
import streamlit as st

DEFAULT_API_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
TIMEOUT_SECONDS = 20


def _fetch_json(base_url: str, endpoint: str, ticker: str) -> dict[str, Any]:
    query = parse.urlencode({"ticker": ticker})
    url = f"{base_url.rstrip('/')}/{endpoint}?{query}"
    req = request.Request(url, method="GET")
    with request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
        payload = response.read().decode("utf-8")
        return json.loads(payload)


def _format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _format_price(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


st.set_page_config(page_title="Terminal", page_icon=":bar_chart:", layout="wide")
st.markdown(
    """
    <style>
      .terminal-title {
        font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
        font-weight: 600;
        font-size: 2.4rem;
        letter-spacing: 0.06rem;
        margin-bottom: 0.1rem;
      }
      .terminal-subtitle {
        color: #6b7280;
        margin-top: 0;
        margin-bottom: 1.4rem;
      }
      .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
      }
    </style>
    <h1 class="terminal-title">Terminal</h1>
    <p class="terminal-subtitle">Stock analytics dashboard</p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    api_url = st.text_input("FastAPI URL", value=DEFAULT_API_URL)
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    load_clicked = st.button("Load analytics", type="primary", use_container_width=True)

if not load_clicked:
    st.info("Choose a ticker, then click **Load analytics**.")
    st.stop()

if not ticker:
    st.error("Ticker is required.")
    st.stop()

responses: dict[str, dict[str, Any]] = {}
errors: dict[str, str] = {}

for endpoint in ("stats", "indicators", "predict"):
    try:
        responses[endpoint] = _fetch_json(api_url, endpoint, ticker)
    except error.HTTPError as exc:
        try:
            detail_payload = json.loads(exc.read().decode("utf-8"))
            detail_message = detail_payload.get("detail", str(detail_payload))
        except json.JSONDecodeError:
            detail_message = exc.reason
        errors[endpoint] = f"HTTP {exc.code}: {detail_message}"
    except error.URLError as exc:
        errors[endpoint] = f"Connection error: {exc.reason}"
    except json.JSONDecodeError:
        errors[endpoint] = "Invalid JSON response from backend."

status_rows = []
for endpoint in ("stats", "indicators", "predict"):
    if endpoint in errors:
        status_rows.append({"endpoint": f"/{endpoint}", "status": "error", "message": errors[endpoint]})
    else:
        status_rows.append({"endpoint": f"/{endpoint}", "status": "ok", "message": "Loaded"})

st.subheader("Endpoint status")
st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

if all(endpoint in errors for endpoint in ("stats", "indicators", "predict")):
    st.error("All endpoint calls failed. Check backend availability and API URL.")
    st.stop()

if "stats" in responses:
    stats_payload = responses["stats"]
    st.subheader(f"{stats_payload.get('ticker', ticker)} summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recent price", _format_price(stats_payload.get("recent_price")))
    c2.metric("Highest price", _format_price(stats_payload.get("highest_price")))
    c3.metric("Lowest price", _format_price(stats_payload.get("lowest_price")))
    c4.metric("Total return", _format_pct(stats_payload.get("total_return")))
    c5.metric("Annualized return", _format_pct(stats_payload.get("annualized_return")))

if "indicators" in responses:
    indicators_payload = responses["indicators"]
    st.subheader("Technical indicators")
    latest = indicators_payload.get("latest", {})
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("MA50", _format_price(latest.get("ma50")))
    i2.metric("Daily return", _format_pct(latest.get("daily_return")))
    i3.metric("Volatility 50", _format_pct(latest.get("volatility50")))
    i4.metric("Volatility 200", _format_pct(latest.get("volatility200")))

    recent_rows = indicators_payload.get("recent", [])
    if recent_rows:
        recent_df = pd.DataFrame(recent_rows)
        st.caption("Recent indicator rows")
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
        if "date" in recent_df.columns and "close" in recent_df.columns:
            close_trend = recent_df[["date", "close"]].copy()
            close_trend["date"] = pd.to_datetime(close_trend["date"])
            close_trend = close_trend.set_index("date")
            st.line_chart(close_trend, use_container_width=True)

if "predict" in responses:
    predict_payload = responses["predict"]
    st.subheader("Next-day prediction")
    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted close", _format_price(predict_payload.get("predicted_next_day_close")))
    metrics = predict_payload.get("metrics", {})
    p2.metric("MAE", f"{metrics.get('mae', 0):.4f}" if "mae" in metrics else "N/A")
    p3.metric("R²", f"{metrics.get('r2', 0):.4f}" if "r2" in metrics else "N/A")

with st.expander("Raw API responses"):
    st.json(responses)
