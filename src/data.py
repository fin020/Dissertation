import yfinance as yf
import pandas as pd

data = yf.download("^GSPC",
                   start="2005-01-01",
                   end="2025-01-01",
                   interval="1d")


pd.DataFrame(data)
print(data.head())
print(data.tail())

data.to_csv("SPY_data.csv")

