import yfinance as yf
import matplotlib.pyplot as plt

data = yf.download("AAPL", start= "2024-01-01", end="2024-11-12")
total_Return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
data['MA50'] = data['Close'].rolling(window=50).mean()
print(total_Return)


plt.figure(figsize=(10,5))
plt.plot(data.index, data['Close']['AAPL'], label = 'AAPL Closing price', color='blue')
plt.plot(data.index, data['Open']['AAPL'], label = 'AAPL Opening price', color='green')
plt.plot(data.index, data['MA50'], label = 'Moving Average AAPL', color='red')
plt.title("Apple (AAPL) Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.show()