import yfinance as yf
import matplotlib.pyplot as plt

# Download data
ticker = input ("Enter Stock Ticker: ").upper()

data = yf.download(ticker, start="2024-01-01", end="2024-11-12")


# Indicators
data['MA50'] = data['Close'].rolling(window=50).mean()
data['Daily Return'] = data['Close'].pct_change()
data['Volatility50'] = data['Daily Return'].rolling(window=50).std()
data['Volatility200'] = data['Daily Return'].rolling(window=200).std()




# --- Step 3: Handle MultiIndex for printing stats ---
def get_numeric_column(col):
    # If column has multiple levels (MultiIndex), select first numeric column
    return col.iloc[:,0] if hasattr(col, 'columns') else col

# Total return
close_col = get_numeric_column(data['Close'])
total_return = (close_col.iloc[-1] - close_col.iloc[0]) / close_col.iloc[0]

high_price = get_numeric_column(data['High']).max()
low_price = get_numeric_column(data['Low']).min()
recent_price = get_numeric_column(data['Close']).iloc[-1]

# --- Step 4: Print key stats ---
print(f"\n--- {ticker} Stock Summary ---")
print(f"Highest Price: ${high_price:.2f}")
print(f"Lowest Price: ${low_price:.2f}")
print(f"Most Recent Price: ${recent_price:.2f}")
print(f"Total Return: {total_return*100:.2f}%")




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

