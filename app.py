from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import streamlit as st
from keras.models import load_model
import datetime
import yfinance as yf 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
yf.pdr_override()

st.title('Stock Trend Prediction')
input = st.text_input('Enter Stock Ticker','GOOG')
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
df = pdr.get_data_yahoo(input, start="2010-01-01", end=current_date)

st.subheader("Today's Stock Price:")
st.write(df["Close"].iloc[-1])


# Load your trained model
model = load_model('model.h5')

# Define the number of time steps (sequence length) used during training
time_steps = 100

# Create a scaler for feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))

# Extract the last 300 values in the 'Close' column for prediction
close_prices = df['Close'].iloc[-3000:].values.reshape(-1, 1)

# Scale the historic data
scaled_close_prices = scaler.fit_transform(close_prices)

# Prepare input features for the model
X_test = []
y_test = []
for i in range(time_steps, len(scaled_close_prices)):
    today_features_scaled = scaled_close_prices[i - time_steps: i].reshape(-1, 1)
    
    # Append to test data
    X_test.append(today_features_scaled)
    y_test.append(scaled_close_prices[i, 0])

X_test = np.array(X_test)
y_test = np.array(y_test)

# Reshape X_test for LSTM input
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform the predictions and test data
y_pred_original_scale = scaler.inverse_transform(y_pred)
y_test_original_scale = scaler.inverse_transform(y_test.reshape(-1, 1))

# Convert y_pred_original_scale and y_test_original_scale to pandas DataFrame for comparison

mse = mean_squared_error(y_test_original_scale, y_pred_original_scale)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_original_scale, y_pred_original_scale)




st.subheader("Today's Predicted Stock Price:")
st.write(y_pred_original_scale[-1, 0])

comparison_df = pd.DataFrame({
    'Actual': y_test_original_scale[-10:, 0],
    'Predicted': y_pred_original_scale[-10:, 0]
})

comparison_df = comparison_df[::-1]
st.subheader("Comparison of Last 10 Predictions and Actual Values:")
st.write(comparison_df)

st.subheader('Date from 2010 - 2023')
st.write(df.tail())

st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100MA")
ma100= df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader("Closing Price vs Time Chart with 100MA & 200MA")
ma100= df.Close.rolling(100).mean()
ma200= df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')

plt.plot(df.Close, 'b')
st.pyplot(fig)


y_pred_original_scale = scaler.inverse_transform(y_pred)
y_test_original_scale = scaler.inverse_transform(y_test.reshape(-1, 1))

st.subheader('Predictions vs Orignal')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test_original_scale,'b',label='Orignal price')
plt.plot(y_pred_original_scale,'r',label = 'predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.subheader("Evaluation Matrix ")
st.write("RMSE = ",rmse)
st.write("MAE  = ",mae)