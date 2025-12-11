import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from dataCollection import collect

def create_dataset(dataset, time_step=60):
	dataX, dataY = [], []
	for i in range(len(dataset) - time_step):
		a = dataset[i:(i + time_step), 0]
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

	data = df.filter(['Close'])
	dataset = data.values

	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled_data = scaler.fit_transform(dataset)

	training_data_len = int(np.ceil(len(dataset) * .8))

	train_data = scaled_data[0:training_data_len, :]
	time_step = 60
	x_train, y_train = create_dataset(train_data, time_step)

	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

	model = Sequential()
	model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
	model.add(LSTM(50, return_sequences=False))
	model.add(Dense(25))
	model.add(Dense(1))

	model.compile(optimizer='adam', loss='mean_squared_error')

	model.fit(x_train, y_train, batch_size=1, epochs=1)

	test_data = scaled_data[training_data_len - time_step:, :]
	x_test, y_test = create_dataset(test_data, time_step)
	
	y_test_actual = dataset[training_data_len:, :]

	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

	predictions = model.predict(x_test)
	predictions = scaler.inverse_transform(predictions)

	train = data[:training_data_len]
	valid = data[training_data_len:]
	valid['Predictions'] = predictions

	plt.figure(figsize=(16, 8))
	plt.title('Model')
	plt.xlabel('Date', fontsize=18)
	plt.ylabel('Close Price USD ($)', fontsize=18)
	plt.plot(train['Close'])
	plt.plot(valid[['Close', 'Predictions']])
	plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
	plt.savefig('prediction.png')
	plt.show()
