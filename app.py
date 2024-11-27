import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os

#Analysis App
st.title("Time Series Analysis")
st.write("Analyze and forecast your time series data with built-in decomposition and ARIMA modeling.")

st.sidebar.info("""
**Instructions:**
1. Upload your dataset or use the built-in sample.
2. Select a column for analysis.
""")

#Selecting Upload or Demo mode
st.sidebar.title("Data Selection")
data_choice = st.sidebar.radio(
    "Choose Data Source:",
    ("Upload Your Own Data", "Use Sample Dataset")
)

#idk this just fixed things
column = 0

#logic based on mode selected
if data_choice == "Use Sample Dataset":
   
    st.write("Using sample dataset.")
    file_path = os.path.join(
    os.getcwd(), "better_sample_data.csv")
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    inuse = 1
else:
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    inuse = 1
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
        else:
            data = pd.read_excel(uploaded_file, parse_dates=['Date'], index_col='Date')
        st.write("Dataset Preview:")
        st.dataframe(data.head())

#choosing data to analyze
if 'data' in locals():
    column = st.selectbox("Select a column to analyze", data.columns)
    if column:
        st.line_chart(data[column])

#line chart
if column:
    st.write(f"Line Chart of {column}:")
    st.line_chart(data[column])

from statsmodels.tsa.seasonal import seasonal_decompose

#trend decomp
if inuse == 1 and column:
    st.subheader("Trend Decomposition")
    
   
    if pd.api.types.is_numeric_dtype(data[column]):
        seasonal_period = st.number_input("Enter the seasonal period (e.g., 365 for yearly seasonality):", min_value=2, value=30)
       
        decomposition = seasonal_decompose(data[column], model='additive', period=seasonal_period)
        
       
        st.write("Original Time Series:")
        st.line_chart(data[column])

        st.write("Trend Component:")
        st.line_chart(decomposition.trend)

        st.write("Seasonal Component:")
        st.line_chart(decomposition.seasonal)

        st.write("Residual Component:")
        st.line_chart(decomposition.resid)
    else:
        st.error("The selected column must be numeric for decomposition.")

#ARIMA
if "data" in locals() and column:
    st.subheader("Forecasting")

    if pd.api.types.is_numeric_dtype(data[column]):
        st.write("Input Forecasting Parameters:")
        p = st.number_input("AR Order (p):", min_value=0, max_value=5, value=1, step=1)
        d = st.number_input("Differencing Order (d):", min_value=0, max_value=2, value=1, step=1)
        q = st.number_input("MA Order (q):", min_value=0, max_value=5, value=1, step=1)
        steps = st.number_input("Number of Steps to Forecast:", min_value=1, max_value=100, value=10, step=1)

        try:
            st.write("Running ARIMA model...")
            model = ARIMA(data[column], order=(p, d, q))
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=steps)
            st.write("Forecast Results:")
            st.write(forecast)

            st.line_chart(pd.concat([data[column], forecast.rename("Forecast")]))
        except Exception as e:
            st.error(f"Error in ARIMA modeling: {e}")
    else:
        st.error("The selected column must be numeric for forecasting.")

if "data" not in locals() or data.empty:
    st.error("No data available. Please upload a dataset or select the sample.")



# Print the current working directory
st.write("Current working directory:", os.getcwd())
