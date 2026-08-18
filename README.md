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
- Streamlit dashboard frontend (**Terminal**) for analytics visualization

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
Backend: FastAPI  
Frontend: Streamlit  

---

## How to Run

### Docker (recommended)

1. Start Docker Desktop (or Docker Engine).
2. From the project root, run:

```bash
docker compose up --build
```

3. Open:
   - Streamlit dashboard: `http://localhost:8501`
   - FastAPI docs: `http://localhost:8000/docs`

---

### Local Python environment

1. Install dependencies

pip install -r requirements.txt

2. Start Redis (required for cached `/predict` endpoint)

3. Start FastAPI backend

uvicorn Stocks_Full_Stack.main:app --reload

4. Start the Streamlit dashboard

streamlit run Stocks_Full_Stack/Dashboard.py

5. Open the dashboard in your browser and enter a stock ticker (e.g. AAPL, TSLA, MSFT)

Optional (CLI-only workflow): run `python stock_analysis.py` for terminal analysis output.

---

## Example Output

- Stock summary statistics
- Price and moving average charts
- Daily return visualization
- Volatility comparison
- Machine learning performance metrics

---

## API Endpoints

- /stats?ticker=AAPL
- /indicators?ticker=TSLA
- /predict?ticker=MSFT

These endpoints power the Streamlit dashboard.

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
