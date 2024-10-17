import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta

# Function to retrieve stock data and earnings
def get_stock_data(ticker):
    # Download historical stock data
    stock_data = yf.download(ticker, period="2y", interval="1d")

    # Fetch earnings data for the stock (This could return None if no data is available)
    ticker_obj = yf.Ticker(ticker)
    earnings_data = ticker_obj.earnings
    
    # Check if earnings_data is None or empty
    if earnings_data is None or earnings_data.empty:
        st.error(f"No earnings data available for {ticker}. Please try another stock.")
        return stock_data, None

    return stock_data, earnings_data

# Function to train a simple model on earnings and stock price change
def train_model(earnings_df):
    # Use percent change in stock price as the target and earnings as the feature
    earnings_df['Price Change %'] = earnings_df['Close'].pct_change() * 100
    earnings_df = earnings_df.dropna()

    X = earnings_df[['Earnings']]  # Feature
    y = earnings_df['Price Change %']  # Target

    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to forecast stock price change based on user-provided earnings
def forecast_stock_price(model, earnings_input):
    predicted_change = model.predict([[earnings_input]])
    return predicted_change[0]  # Returns predicted price change percentage

# Streamlit App: Earnings Forecasting Page
def display_earnings_forecast_page():
    st.title("Earnings-Based Stock Price Forecasting")

    # User inputs stock symbol and tests different earnings
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", "")
    test_earnings = st.number_input("Enter Earnings Estimate", value=0.0, format="%.2f")

    if st.button("Run Earnings Forecast"):
        if stock_symbol:
            # Fetch historical data and earnings for the stock
            stock_data, earnings_data = get_stock_data(stock_symbol)

            # Ensure earnings_data is not None
            if earnings_data is not None:
                # Join stock data and earnings data on the dates
                merged_data = stock_data[['Close']].merge(earnings_data, left_index=True, right_index=True, how='inner')

                # Train a model on the historical earnings vs price change
                model = train_model(merged_data)

                # Use the model to predict price change for the input earnings
                forecasted_price_change = forecast_stock_price(model, test_earnings)

                # Display the forecasted change
                st.write(f"Forecasted Stock Price Change: {forecasted_price_change:.2f}% based on earnings estimate of {test_earnings}")
            else:
                st.error("Unable to fetch earnings data for the selected stock.")
        else:
            st.error("Please enter a valid stock symbol.")

# Call the display function
if __name__ == "__main__":
    display_earnings_forecast_page()
