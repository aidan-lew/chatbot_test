# streamlit_app.py
import streamlit as st
from earnings_summarization import display_earnings_page
from forecasting_page import display_forecasting_page
from earnings_forecast import display_earnings_forecast_page

# Set up the page layout and basic styles
st.set_page_config(page_title="Corporate Forecasting App", layout="wide")

# Custom CSS for better layout
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .api-input {
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Add a sidebar for navigation
page = st.sidebar.selectbox("Select a Page", ("Forecasts", "Earnings", "Report Summarizer"))

# Call the appropriate function based on the selected page
if page == "Report Summarizer":
    display_earnings_page()  # Call the earnings summarization function
elif page == "Forecasts":
    display_forecasting_page()  # Call the forecasting function
elif page == "Earnings":
    display_earnings_forecast_page() 
