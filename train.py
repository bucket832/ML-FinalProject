import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from dataCollection import collect
from visualization import visualizePCA

def add_technical_indicators(df):
	df['SMA_50'] = df['Close'].rolling(window=50).mean()
	
	delta = df['Close'].diff()
	gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
	loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
	rs = gain / loss
	df['RSI'] = 100 - (100 / (1 + rs))
	
	exp1 = df['Close'].ewm(span=12, adjust=False).mean()
	exp2 = df['Close'].ewm(span=26, adjust=False).mean()
	df['MACD'] = exp1 - exp2
	df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
	return df

def create_dataset(dataset, target, time_step=60):
	dataX, dataY = [], []
	for i in range(len(dataset) - time_step):
		a = dataset[i:(i + time_step), :]
		dataX.append(a)
		dataY.append(target[i + time_step])
	return np.array(dataX), np.array(dataY)

if __name__ == "__main__":
	ticker = input("Enter stock ticker (e.g. AAPL, TSLA): ").strip().upper()
	if not ticker:
		print("No ticker entered, defaulting to AAPL.")
		ticker = "AAPL"
	
	try:
		df = collect(ticker)
		sp500 = collect('^GSPC')
	except Exception as e:
		print(f"Error collecting data: {e}")
		exit(1)

	sp500 = sp500[['Close']].rename(columns={'Close': 'SP500_Close'})
	df = df.join(sp500, how='left')
	
	df = add_technical_indicators(df)
	df = df.dropna()
	
	features = ['Close', 'SMA_50', 'RSI', 'MACD', 'Signal_Line', 'SP500_Close']
	data = df[features].values
	target = df[['Close']].values
	
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled_data = scaler.fit_transform(data)
	
	target_scaler = MinMaxScaler(feature_range=(0, 1))
	scaled_target = target_scaler.fit_transform(target)
	
	training_data_len = int(np.ceil(len(data) * .8))
	
	train_data = scaled_data[0:training_data_len, :]
	train_target = scaled_target[0:training_data_len]
	
	time_step = 60
	x_train, y_train = create_dataset(train_data, train_target, time_step)
	
	model = Sequential()
	model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
	model.add(LSTM(50, return_sequences=False))
	model.add(Dense(25))
	model.add(Dense(1))
	
	model.compile(optimizer='adam', loss='mean_squared_error')
	model.fit(x_train, y_train, batch_size=1, epochs=1)
	
	test_data = scaled_data[training_data_len - time_step:, :]
	test_target = scaled_target[training_data_len - time_step:]
	
	x_test, y_test = create_dataset(test_data, test_target, time_step)
	
	predictions = model.predict(x_test)
	predictions = target_scaler.inverse_transform(predictions)
	
	valid = df[training_data_len:].copy()
	valid['Predictions'] = predictions

	rmse = np.sqrt(mean_squared_error(valid['Close'], valid['Predictions']))
	mape = np.mean(np.abs((valid['Close'] - valid['Predictions']) / valid['Close'])) * 100
	print(f"Root Mean Squared Error (RMSE): {rmse}")
	print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
	
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
	
	ax1.set_title(f'Model Prediction for {ticker}')
	ax1.set_ylabel('Close Price USD ($)')
	ax1.plot(df.index[:training_data_len], df['Close'][:training_data_len], label='Train')
	ax1.plot(valid.index, valid['Close'], label='Val')
	ax1.plot(valid.index, valid['Predictions'], label='Predictions')
	ax1.legend(loc='lower right')
	
	ax2.set_ylabel('RSI')
	ax2.plot(df.index[:training_data_len], df['RSI'][:training_data_len], label='Train RSI', alpha=0.3)
	ax2.plot(valid.index, valid['RSI'], label='Val RSI', color='purple')
	ax2.axhline(70, linestyle='--', alpha=0.5, color='red')
	ax2.axhline(30, linestyle='--', alpha=0.5, color='green')
	ax2.legend(loc='lower right')
	
	ax3.set_ylabel('MACD')
	ax3.plot(df.index[:training_data_len], df['MACD'][:training_data_len], label='Train MACD', alpha=0.3)
	ax3.plot(valid.index, valid['MACD'], label='Val MACD')
	ax3.plot(valid.index, valid['Signal_Line'], label='Val Signal')
	ax3.legend(loc='lower right')
	
	plt.xlabel('Date')
	plt.savefig(f'{ticker}_prediction.png')
	plt.show()
	
	last_60_days = scaled_data[-60:]
	X_test_next = np.array([last_60_days])
	X_test_next = np.reshape(X_test_next, (X_test_next.shape[0], X_test_next.shape[1], X_test_next.shape[2]))
	pred_price = model.predict(X_test_next)
	pred_price = target_scaler.inverse_transform(pred_price)
	print(f"Predicted close price for next day for {ticker}: {pred_price[0][0]}")
