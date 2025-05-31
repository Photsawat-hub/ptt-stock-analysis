import yfinance as yf
import pandas as pd

df = yf.download("PTT.BK", start="2024-05-01", end="2025-05-31", auto_adjust=False)

df.reset_index(inplace=True)

df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

df.to_csv("ptt_daily_price.csv", index=False)
