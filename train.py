import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, SimpleRNN
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, Callback
from dataCollection import collect

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
		self.weights = []

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		# Get weights from the first dense layer (index -2)
		# We take the mean of weights to simplify visualization to 1 dimension per layer/neuron
		w = self.model.layers[-2].get_weights()[0] 
		self.weights.append(np.mean(w))

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

def build_baseline_model(input_shape):
	"""Simple RNN model to serve as a baseline."""
	model = Sequential()
	model.add(SimpleRNN(50, input_shape=input_shape, return_sequences=False))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mean_squared_error')
	return model

def build_advanced_model(input_shape):
	"""Bidirectional LSTM with Dropout and Regularization."""
	model = Sequential()
	model.add(Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=l2(0.001)), input_shape=input_shape))
	model.add(Dropout(0.2))
	model.add(Bidirectional(LSTM(50, return_sequences=False, kernel_regularizer=l2(0.001))))
	model.add(Dropout(0.2))
	model.add(Dense(25, activation='relu'))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mean_squared_error')
	return model

def plot_prediction_surface(model, last_sequence, scaler, target_scaler):
	"""
	Plots a 3D surface showing how the model's prediction changes 
	with respect to Price and RSI variations in the last time step.
	"""
	base_seq = last_sequence[0].copy() # (60, 6)
	
	# Create ranges for Close Price (feature 0) and RSI (feature 2)
	current_close = base_seq[-1, 0]
	current_rsi = base_seq[-1, 2]
	
	close_range = np.linspace(current_close * 0.9, current_close * 1.1, 20)
	rsi_range = np.linspace(max(0, current_rsi - 20), min(100, current_rsi + 20), 20)
	
	X_mesh, Y_mesh = np.meshgrid(close_range, rsi_range)
	
	# Prepare batch of inputs
	batch_inputs = []
	for i in range(X_mesh.shape[0]):
		for j in range(X_mesh.shape[1]):
			seq = base_seq.copy()
			seq[-1, 0] = X_mesh[i, j]
			seq[-1, 2] = Y_mesh[i, j]
			batch_inputs.append(seq)
			
	batch_inputs = np.array(batch_inputs)
	
	# Predict
	predictions = model.predict(batch_inputs, verbose=0)
	predictions = target_scaler.inverse_transform(predictions)
	
	Z_mesh = predictions.reshape(X_mesh.shape)
	
	fig = plt.figure(figsize=(12, 8))
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='viridis')
	
	ax.set_xlabel('Normalized Close Price (t)')
	ax.set_ylabel('Normalized RSI (t)')
	ax.set_zlabel('Predicted Price (t+1)')
	ax.set_title('Prediction Surface: Price vs RSI vs Prediction')
	fig.colorbar(surf)
	plt.savefig('prediction_surface_3d.png')
	plt.show()

def plot_loss_landscape(history):
	"""
	Plots the Loss Landscape showing how the loss changes as weights evolve.
	X-axis: Epochs
	Y-axis: Mean Weight Value
	Z-axis: Loss
	"""
	epochs = range(1, len(history.losses) + 1)
	weights = history.weights
	losses = history.losses
	
	fig = plt.figure(figsize=(12, 8))
	ax = fig.add_subplot(111, projection='3d')
	
	# Create a path line
	ax.plot(epochs, weights, losses, color='red', marker='o', linewidth=2, label='Training Path')
	
	# Create a surface for context (optional, using triangulation)
	ax.plot_trisurf(epochs, weights, losses, cmap='viridis', alpha=0.6)
	
	ax.set_xlabel('Epochs')
	ax.set_ylabel('Mean Weight Value')
	ax.set_zlabel('Loss')
	ax.set_title('Loss Landscape: Evolution of Weights & Loss')
	plt.legend()
	plt.savefig('loss_landscape_3d.png')
	plt.show()

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
	
	# --- Baseline Model ---
	print("\nTraining Baseline Model (Simple RNN)...")
	baseline_model = build_baseline_model((x_train.shape[1], x_train.shape[2]))
	baseline_model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)
	
	# --- Advanced Model ---
	print("\nTraining Advanced Model (Bi-LSTM + Dropout)...")
	model = build_advanced_model((x_train.shape[1], x_train.shape[2]))
	
	loss_history = LossHistory()
	early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
	model.fit(x_train, y_train, batch_size=32, epochs=50, callbacks=[early_stopping, loss_history], verbose=1)
	
	test_data = scaled_data[training_data_len - time_step:, :]
	test_target = scaled_target[training_data_len - time_step:]
	
	x_test, y_test = create_dataset(test_data, test_target, time_step)
	
	# Evaluate Baseline
	baseline_preds = baseline_model.predict(x_test)
	baseline_preds = target_scaler.inverse_transform(baseline_preds)
	baseline_rmse = np.sqrt(mean_squared_error(df['Close'][training_data_len:], baseline_preds))
	print(f"Baseline Model RMSE: {baseline_rmse}")

	# Evaluate Advanced
	predictions = model.predict(x_test)
	predictions = target_scaler.inverse_transform(predictions)
	
	valid = df[training_data_len:].copy()
	valid['Predictions'] = predictions

	rmse = np.sqrt(mean_squared_error(valid['Close'], valid['Predictions']))
	mape = np.mean(np.abs((valid['Close'] - valid['Predictions']) / valid['Close'])) * 100
	print(f"Advanced Model RMSE: {rmse}")
	print(f"Advanced Model MAPE: {mape}%")
	
	# Plot 3D Surface
	last_60_days = scaled_data[-60:]
	X_test_next = np.array([last_60_days])
	X_test_next = np.reshape(X_test_next, (X_test_next.shape[0], X_test_next.shape[1], X_test_next.shape[2]))
	
	plot_prediction_surface(model, X_test_next, scaler, target_scaler)
	plot_loss_landscape(loss_history)
	
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
	
	ax1.set_title(f'Model Prediction for {ticker}')
	ax1.set_ylabel('Close Price USD ($)')
	ax1.plot(df.index[:training_data_len], df['Close'][:training_data_len], label='Train')
	ax1.plot(valid.index, valid['Close'], label='Val')
	ax1.plot(valid.index, valid['Predictions'], label='Predictions')
	ax1.legend(loc='lower right')
	
	ax2.set_ylabel('RSI')
	ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
	ax2.axhline(70, linestyle='--', alpha=0.5, color='red')
	ax2.axhline(30, linestyle='--', alpha=0.5, color='green')
	ax2.legend(loc='lower right')
	
	ax3.set_ylabel('MACD')
	ax3.plot(df.index, df['MACD'], label='MACD')
	ax3.plot(df.index, df['Signal_Line'], label='Signal')
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
