import datetime
import pandas as pd
import yfinance as yf

PERIOD = "5y"

def collect(ticker: str) -> pd.DataFrame:
	end = datetime.date.today()
	start = end - datetime.timedelta(days=5 * 365)

	data = yf.download(ticker, period=PERIOD)

	if isinstance(data.columns, pd.MultiIndex):
		data.columns = data.columns.get_level_values(0)

	if data is None or data.empty:
		raise ValueError(f"No data returned for ticker: {ticker}")

	expected_cols = ["Open", "Close", "High", "Low", "Volume"]
	missing = [c for c in expected_cols if c not in data.columns]
	if missing:
		raise ValueError(f"Missing expected columns in returned data: {missing}")

	df = data[expected_cols].copy()
	df.index = pd.to_datetime(df.index)
	return df

