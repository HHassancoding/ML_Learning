# Stock Analysis & Price Prediction Platform

A full-stack-ready stock analysis project that retrieves real market data, computes financial indicators, and uses machine learning to predict future stock prices.

---

## Features

- Real-time stock data using Yahoo Finance
- Technical indicators:
  - 50-day Moving Average
  - Daily Returns
  - Rolling Volatility (50-day and 200-day)
- Performance metrics:
  - Highest price
  - Lowest price
  - Most recent price
  - Total return
  - Annualized return
- Machine Learning:
  - Random Forest Regressor
  - Predicts next-day closing price
  - Evaluated using MAE and R²
- Backend-ready architecture for FastAPI integration

---

## Machine Learning Overview

- Target: Next-day closing price
- Features:
  - Open, High, Low, Close, Volume
  - Moving averages
  - Volatility metrics
- Model: RandomForestRegressor
- Evaluation Metrics:
  - Mean Absolute Error (MAE)
  - R² Score

This model learns patterns from historical data rather than memorizing prices.

---

## Tech Stack

Data: Python, pandas, numpy, yfinance  
Machine Learning: scikit-learn  
Visualization: matplotlib  
Backend (in progress): FastAPI  
Frontend (planned): React / Next.js  

---

## How to Run

1. Install dependencies

pip install -r requirements.txt

2. Run the project

python stock_analysis.py

3. Enter a stock ticker when prompted (e.g. AAPL, TSLA, MSFT)

---

## Example Output

- Stock summary statistics
- Price and moving average charts
- Daily return visualization
- Volatility comparison
- Machine learning performance metrics

---

## API Integration (In Progress)

Planned FastAPI endpoints:

- /stats?ticker=AAPL
- /indicators?ticker=TSLA
- /predict?ticker=MSFT

These endpoints will return JSON for frontend dashboards.

---

## Future Improvements

- Interactive web dashboard
- Multi-stock comparison
- Model persistence
- Improved feature engineering
- Cloud deployment

---

## Purpose of This Project

This project demonstrates:
- Real-world data handling
- Applied machine learning
- Financial domain understanding
- Backend and full-stack readiness

Built to reflect industry-level practices rather than toy examples.
