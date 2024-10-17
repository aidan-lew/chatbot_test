import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.callbacks import EarlyStopping
import os
import tensorflow 

# Function to download stock data and calculate technical indicators
def get_numeric_data(ticker):
    period = '2y'
    interval = "1h"

    # Retrieve the historical market data
    data = yf.download(tickers=ticker, period=period, interval=interval)

    # Calculate moving averages and other technical indicators
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()

    # Calculate RSI
    delta = data['Close'].diff(1)
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean().abs()
    RS = roll_up / roll_down
    data['RSI'] = 100.0 - (100.0 / (1.0 + RS))

    # Return a dataframe with numeric values
    numeric_df = data.select_dtypes(include=['number'])
    numeric_df.reset_index(inplace=True)
    return numeric_df

# Function to train the GRU model and save it
def train_gru_forecast(target, df, save_path):
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    close_df = df[[target]]

    scaler = StandardScaler()
    scaled_close = scaler.fit_transform(close_df)

    seq_len = 1
    X_train = []
    y_train = []
    for i in range(seq_len, len(scaled_close)):
        X_train.append(scaled_close[i-seq_len:i])
        y_train.append(scaled_close[i])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(GRU(units=1000, return_sequences=True, input_shape=(seq_len, 1)))
    model.add(Dropout(0.26))
    model.add(GRU(units=200, return_sequences=True))
    model.add(Dropout(0.26))
    model.add(GRU(units=1000, return_sequences=False))
    model.add(Dropout(0.26))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='loss', patience=5, min_delta=0.01)
    
    # Add a Streamlit progress bar
    progress_bar = st.progress(0)

    # Train the model and update progress in Streamlit
    for epoch in range(200):  # Limit to 200 epochs
        history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)
        progress_bar.progress((epoch + 1) / 200)
        if epoch > 5 and history.history['loss'][-1] < 0.01:
            break

    # Save the trained model
    model.save(save_path)
    st.success(f"Model saved to {save_path}")

# Streamlit app to train and save the model
st.title("Train GRU Model for Stock Forecasting")

# User input for stock symbol and trigger model training
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", "")
if st.button("Train GRU Model"):
    if stock_symbol:
        # Get numeric data for the stock
        st.write(f"Fetching data for {stock_symbol}...")
        numeric_data = get_numeric_data(stock_symbol)

        # Train and save the GRU model
        st.write("Training the GRU model. This may take a while...")
        save_path = os.path.join(os.getcwd(), 'gru_model.keras')
        train_gru_forecast('Close', numeric_data, save_path)
    else:
        st.error("Please enter a valid stock symbol.")
