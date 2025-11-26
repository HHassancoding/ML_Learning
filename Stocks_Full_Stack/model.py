import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def download_stock(ticker):
    data = yf.download(ticker, start="2025-01-01", end="2025-10-12")
    return data


def add_indicators(data):
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['Daily Return'] = data['Close'].pct_change()
    data['Volatility50'] = data['Daily Return'].rolling(window=50).std()
    data['Volatility200'] = data['Daily Return'].rolling(window=200).std()
    return data


def prepare_training_data(data):
    data['target'] = data['Close'].shift(-1)   # predict next-day price
    data = data[:-1]

    X = data.drop(columns=['target'])
    y = data['target']

    return train_test_split(X, y, train_size=0.3, random_state=42)


def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
