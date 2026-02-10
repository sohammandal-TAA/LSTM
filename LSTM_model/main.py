from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

data = pd.read_csv('../LSTM_model/MicrosoftStock.csv')

#print(data.head())
# print(data.info())
# print(data.describe())

#Initial EDA
#plot 1 - Open & Close Price

# plt.figure(figsize=(14, 7))
# plt.plot(data['date'], data['open'], label='Open Price', color='blue')
# plt.plot(data['date'], data['close'], label='Close Price', color='red')
# plt.title('Open and Close Prices over time')
# plt.xlabel('Date')
# plt.ylabel('Price (USD)')
# plt.legend()
# plt.tight_layout()
# plt.show()

# #plot 2 - Volume
# plt.figure(figsize=(14, 7))
# plt.plot(data['date'], data['volume'], label='Volume', color='green')
# plt.title('Stock Volume over time')
# plt.show()

# Drop non-numeric columns

numeric_data = data.select_dtypes(include=["int64", "float64"])
# print(numeric_data.head())

# Plot 3 - Correlation Heatmap

# plt.figure(figsize=(10, 8))
# sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Heatmap of Stock Features')
# plt.show()

#Convert date to datetime and create a date filter
data['date'] = pd.to_datetime(data['date'])

prediction_data = data.loc[
    (data['date'] > datetime(2013,1,1)) &
    (data['date'] < datetime(2018,1,1))
]

# plt.figure(figsize=(14, 7))
# plt.plot(data['date'], data['close'], color='blue')
# plt.xlabel('Date')
# plt.ylabel('Close')
# plt.title('Price over time')

# Prepare for the LSTM model (Sequence)

stock_close = data.filter(['close'])

dataset = stock_close.values
training_data_len = int(np.ceil( len(dataset) * .95 ))

#Preprocess the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[0:training_data_len] #95% of all out data

X_train, y_train = [], []

#creating a sliding window of 60 days to predict the next day   
for i in range(60, len(training_data)):
    X_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) #[0] is the number of samples (first element), [1] is the number of features 

#Build the LSTM model
model = keras.Sequential()

#First Layer
model.add(keras.layers.LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))

#Second Layer
model.add(keras.layers.LSTM(128, return_sequences=False))

#Third Layer
model.add(keras.layers.Dense(256, activation='relu'))

#Fourth Layer (Dropout) for overfitting
model.add(keras.layers.Dropout(0.5))

#Final output Layer
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[keras.metrics.RootMeanSquaredError()])

training = model.fit(X_train, y_train, batch_size=32, epochs=30)

#Prep the test data
test_data = scaled_data[training_data_len - 60:]
X_test, y_test = [], dataset[training_data_len:]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

#Plot the data
train = data[:training_data_len]
test = data[training_data_len:]
test = test.copy()

test['Predictions'] = predictions
plt.figure(figsize=(14, 7))
plt.plot(train['date'], train['close'], label='Train (Actual)', color='blue')
plt.plot(test['date'], test['close'], label='Test (Actual)', color='green')
plt.plot(test['date'], test['Predictions'], label='Predictions', color='red')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Microsoft Stock Price Prediction')
plt.legend()
plt.show()