import yfinance as yf
import matplotlib.pyplot as plt
from PIL.EpsImagePlugin import split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Download data
ticker = input ("Enter Stock Ticker: ").upper()

data = yf.download(ticker, start="2025-01-01", end="2025-10-12")


# Indicators
data['MA50'] = data['Close'].rolling(window=50).mean()
data['Daily Return'] = data['Close'].pct_change()
data['Volatility50'] = data['Daily Return'].rolling(window=50).std()
data['Volatility200'] = data['Daily Return'].rolling(window=200).std()
close_col = data['Close'].iloc[:,0] if hasattr(data['Close'], 'columns') else data['Close']





# --- Step 3: Handle MultiIndex for printing stats ---
def get_numeric_column(col):
    # If column has multiple levels (MultiIndex), select first numeric column
    return col.iloc[:,0] if hasattr(col, 'columns') else col

# Total return
close_col = get_numeric_column(data['Close'])
total_return = (close_col.iloc[-1] - close_col.iloc[0]) / close_col.iloc[0]

#Number of trading days
N = len(data)

#Annual return
annual_return = (1 + total_return) ** (252 / N) - 1

high_price = get_numeric_column(data['High']).max()
low_price = get_numeric_column(data['Low']).min()
recent_price = get_numeric_column(data['Close']).iloc[-1]

# --- Step 4: Print key stats ---
print(f"\n--- {ticker} Stock Summary ---")
print(f"Highest Price: ${high_price:.2f}")
print(f"Lowest Price: ${low_price:.2f}")
print(f"Most Recent Price: ${recent_price:.2f}")
print(f"Total Return: {total_return*100:.2f}%")
print(f"Annual Return: {annual_return*100:.2f}%")

data['target'] = data['Close'].shift(-1)
data = data[:-1]
X = data.drop(columns=['target'])
Y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.3, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train,y_train)
prediction = model.predict(X_test)
mean  = mean_absolute_error(y_test,prediction)
r2 = r2_score(y_test, prediction)
print(mean)
print(r2)



# Plot closing price + MA50
plt.figure(figsize=(12,5))
plt.plot(data.index, data['Close'], label='AAPL Closing Price', color='blue')
plt.plot(data.index, data['MA50'], label='MA50', color='orange')
plt.title(f"{ticker} Closing Price & 50-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.show()

#Daily Returns
plt.figure(figsize=(12,5))
plt.plot(data.index, data['Daily Return'], label='Daily Return', color='red')
plt.title(f"{ticker} Daily Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.legend()
plt.show()


#Volatility 50 vs 200
plt.figure(figsize=(12,5))
plt.plot(data.index, data['Volatility50'], label='50-Day Volatility', color='purple')
plt.plot(data.index, data['Volatility200'], label='200-Day Volatility', color='green')
plt.title(f"{ticker} Rolling Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.show()

