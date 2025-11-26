from fastapi import FastAPI
from model import download_stock, add_indicators, prepare_training_data, train_model

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Stock Prediction API is running"}

@app.get("/stock-info/{ticker}")
def stock_info(ticker: str):
    data = download_stock(ticker)
    data = add_indicators(data)
    data = data.dropna()  # clean

    # Basic stats
    high = data['High'].max()
    low = data['Low'].min()
    recent = data['Close'].iloc[-1]

    return {
        "ticker": ticker.upper(),
        "highest_price": float(high),
        "lowest_price": float(low),
        "recent_price": float(recent),
    }


@app.get("/predict/{ticker}")
def predict_price(ticker: str):
    data = download_stock(ticker)
    data = add_indicators(data)
    data = data.dropna()

    X_train, X_test, y_train, y_test = prepare_training_data(data)
    model = train_model(X_train, y_train)

    # Predict the NEXT DAY (last row)
    next_day_features = X_test.iloc[-1]
    pred = model.predict([next_day_features])[0]

    return {
        "ticker": ticker.upper(),
        "predicted_price": float(pred)
    }
