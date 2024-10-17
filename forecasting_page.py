import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load the GRU model from file
model = load_model('gru_model.keras')

def get_numeric_data(ticker):
    period = '6mo'
    interval = "1d"

    # Retrieve the historical market data (last 6 months)
    data = yf.download(tickers=ticker, period=period, interval=interval)
    data.reset_index(inplace=True)
    return data

def gru_forecast(model, df, target, future_days, noise_factor=0.02):
    # Prepare the data for forecasting
    close_df = df[[target]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(close_df)

    # Prepare the input data for the model
    seq_len = 1
    last_seq = scaled_close[-seq_len:]
    last_seq = last_seq.reshape((1, seq_len, 1))

    # Predict future prices
    future_close_prices = []
    for i in range(future_days):
        prediction = model.predict(last_seq)

        # Add randomness to the prediction (Monte Carlo approach)
        noise = np.random.normal(0, noise_factor)
        prediction_with_noise = prediction + noise

        # Reshape prediction to match the 3D shape required for concatenation
        prediction_with_noise = prediction_with_noise.reshape((1, 1, 1))

        # Append the prediction to the last sequence
        last_seq = np.append(last_seq[:, 1:, :], prediction_with_noise, axis=1)

        # Inverse transform the prediction to get the actual price
        future_close_prices.append(scaler.inverse_transform(prediction_with_noise.reshape(-1, 1))[0][0])
    
    return future_close_prices

# Monte Carlo simulations
def monte_carlo_simulations(model, data, target, future_days, num_simulations, noise_factor):
    all_simulations = []
    for _ in range(num_simulations):
        future_prices = gru_forecast(model, data, target, future_days, noise_factor)
        all_simulations.append(future_prices)
    return all_simulations

# Streamlit app
def display_forecasting_page():
    st.title("Stock Price Forecasting")

    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", "")
    forecast_days = st.number_input("Number of Days to Forecast", min_value=1, max_value=365, value=30)
    num_simulations = st.number_input("Number of Simulations", min_value=1, max_value=100, value=10)
    noise_factor = st.slider("Monte Carlo Noise Factor", min_value=0.01, max_value=0.1, value=0.02, step=0.01)

    if st.button("Run Forecast"):
        if stock_symbol:
            # Fetch historical stock data (last 6 months)
            data = get_numeric_data(stock_symbol)
            
            # Run Monte Carlo simulations
            simulations = monte_carlo_simulations(model, data, 'Close', forecast_days, num_simulations, noise_factor)

            # Calculate the average of all simulations
            avg_simulation = np.mean(simulations, axis=0)

            # Plot results
            future_dates = pd.date_range(data['Date'].max() + timedelta(days=1), periods=forecast_days)

            # Create a plot with a black background
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_facecolor('black')
            fig.patch.set_facecolor('black')

            # Plot historical data (last 6 months)
            ax.plot(data['Date'], data['Close'], color='cyan', label='Historical Data')

            # Plot each Monte Carlo simulation in bright colors
            for sim in simulations:
                ax.plot(future_dates, sim, color='magenta', alpha=0.2)

            # Plot the average simulation as a bold yellow line
            ax.plot(future_dates, avg_simulation, color='yellow', label='Average Forecast', linewidth=2)

            # Customize plot appearance
            ax.set_title(f"Price Forecast for {stock_symbol}", color='white')
            ax.set_xlabel("Date", color='white')
            ax.set_ylabel("Price", color='white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(colors='white')
            ax.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')

            # Display the plot
            st.pyplot(fig)
        else:
            st.error("Please enter a valid stock symbol.")

# To display the Streamlit page, call the function
if __name__ == "__main__":
    display_forecasting_page()
