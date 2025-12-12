import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LSTM, Dense
from dataCollection import collect
from visualization import visualizePCA

NUM_DATA_POINTS = 5 #OHLCV

def create_dataset(dataset, time_step=60): # TODO OHLCV for dataset
	dataX, dataY = [], []
	for i in range(len(dataset) - time_step):
		a = dataset[i:(i + time_step), :NUM_DATA_POINTS]
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

if __name__ == "__main__":
	ticker = "AAPL"
	try:
		df = collect(ticker)
	except Exception as e:
		print(f"Error collecting data: {e}")
		exit(1)


	dataset = df.to_numpy()

	training_data_len = int(np.ceil(len(dataset) * .8))
	training_data_raw = dataset[:training_data_len, :]
	testing_data_raw = dataset[training_data_len:, :]




	scaler = MinMaxScaler(feature_range=(0, 1))
	training_data_scaled = scaler.fit_transform(training_data_raw)

	testing_data_scaled = scaler.transform(testing_data_raw)

	close_min = scaler.data_min_[0]
	close_max = scaler.data_max_[0]


	time_step = 60
	x_train, y_train = create_dataset(training_data_scaled, time_step)

	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], NUM_DATA_POINTS))

	model = Sequential()
	model.add(LSTM(50, return_sequences=True,
				input_shape=(x_train.shape[1],
				NUM_DATA_POINTS)))
	model.add(LSTM(50, return_sequences=False))
	model.add(Dense(25))
	model.add(Dense(1))

	# model = Sequential()
	# model.add(LSTM(50, return_sequences=True,
	# 			input_shape=(x_train.shape[1],
	# 			NUM_DATA_POINTS),
	# 			kernel_regularizer=l2(0.001)))
	# model.add(LSTM(50, return_sequences=False, kernel_regularizer=l2(0.001)))
	# model.add(Dense(25, kernel_regularizer=l2(0.001)))
	# model.add(Dense(1))

	model.compile(optimizer='adam', loss='mean_squared_error')

	model.fit(x_train, y_train, batch_size=1, epochs=1)

	x_test, y_test = create_dataset(testing_data_scaled, time_step)
	
	# y_test_actual = dataset[training_data_len:, :]

	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], NUM_DATA_POINTS))

	predictions = model.predict(x_test)
	predictions = predictions * (close_max - close_min) + close_min

	visualizePCA(x_test.reshape(x_test.shape[0], -1), predictions)

	train = df[:training_data_len+time_step]
	valid = df[training_data_len+time_step:]
	valid['Predictions'] = predictions
	test_RMSE = np.sqrt(mean_squared_error(valid['Close'], valid['Predictions']))
	print(f"Testing RMSE = {test_RMSE:.4f}")


	plt.figure(figsize=(16, 8))
	plt.title('Model')
	plt.xlabel('Date', fontsize=18)
	plt.ylabel('Close Price USD ($)', fontsize=18)
	plt.plot(train['Close'])
	plt.plot(valid[['Close', 'Predictions']])
	plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
	plt.savefig('prediction.png')
	plt.show()
