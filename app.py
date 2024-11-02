import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import streamlit as st
from keras.models import load_model
import datetime
import yfinance as yf  # Import yfinance for fetching stock data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Streamlit app title
st.title('Stock Trend Prediction')

# Input for stock ticker
input = st.text_input('Enter Stock Ticker', 'GOOG')
current_date = datetime.datetime.now().strftime("%Y-%m-%d")

# Fetch stock data using yfinance directly
df = yf.download(input, start="2010-01-01", end=current_date)

# Display today's stock price
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

# Calculate evaluation metrics
mse = mean_squared_error(y_test_original_scale, y_pred_original_scale)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_original_scale, y_pred_original_scale)

# Display predicted stock price
st.subheader("Today's Predicted Stock Price:")
st.write(y_pred_original_scale[-1, 0])

# Create a DataFrame for comparison
comparison_df = pd.DataFrame({
    'Actual': y_test_original_scale[-10:, 0],
    'Predicted': y_pred_original_scale[-10:, 0]
})

comparison_df = comparison_df[::-1]  # Reverse for better display
st.subheader("Comparison of Last 10 Predictions and Actual Values:")
st.write(comparison_df)

# Display last few rows of the stock data
st.subheader('Date from 2010 - 2023')
st.write(df.tail())

# Plot Closing Price vs Time Chart
st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

# Plot Closing Price vs Time Chart with 100MA
st.subheader("Closing Price vs Time Chart with 100MA")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, label='100 MA', color='orange')
plt.plot(df.Close, label='Close Price', color='blue')
plt.legend()
st.pyplot(fig)

# Plot Closing Price vs Time Chart with 100MA & 200MA
st.subheader("Closing Price vs Time Chart with 100MA & 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r', label='100 MA')
plt.plot(ma200, 'g', label='200 MA')
plt.plot(df.Close, 'b', label='Close Price')
plt.legend()
st.pyplot(fig)

# Plot Predictions vs Original
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test_original_scale, 'b', label='Original Price')
plt.plot(y_pred_original_scale, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Display Evaluation Metrics
st.subheader("Evaluation Metrics")
st.write("RMSE = ", rmse)
st.write("MAE  = ", mae)
