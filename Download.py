import yfinance as yf

ticker = "PTT.BK"
start_date = "2024-05-01"
end_date = "2024-05-31"

df = yf.download(ticker, start=start_date, end=end_date)

print(df.head())

df.to_csv("ptt_daily_price1.csv")
