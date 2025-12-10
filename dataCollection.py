import datetime
import pandas as pd
import yfinance as yf

PERIOD = "5y"

def collect(ticker: str) -> pd.DataFrame:
	"""Fetch Open/Close/High/Low/Volume (OCHLV) for `ticker` for the last 5 years.
	Args:
		ticker: Stock ticker symbol (e.g. 'AAPL').
	Returns:
		pandas.DataFrame indexed by date with columns ['Open','Close','High','Low','Volume'].
	Raises:
		ValueError: If no data is returned or required columns are missing.
	"""
	end = datetime.date.today()
	# approximate 5 years as 5*365 days
	start = end - datetime.timedelta(days=5 * 365)

	# Use yfinance to download historical data
	data = yf.download(ticker, period=PERIOD)
	print(data[0])

	if data is None or data.empty:
		raise ValueError(f"No data returned for ticker: {ticker}")

	expected_cols = ["Open", "Close", "High", "Low", "Volume"]
	missing = [c for c in expected_cols if c not in data.columns]
	if missing:
		raise ValueError(f"Missing expected columns in returned data: {missing}")

	df = data[expected_cols].copy()
	df.index = pd.to_datetime(df.index)
	return df

