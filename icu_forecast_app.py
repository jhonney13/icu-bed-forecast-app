import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_components_plotly
import plotly.graph_objs as go

# --- Page Config ---
st.set_page_config(page_title="ICU Forecast", layout="wide", initial_sidebar_state="expanded")

# --- Custom Theme Styling ---
st.markdown(
    """
    <style>
    .main {background-color: #f5f6fa;}
    .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Logo/Banner ---
st.title("ICU Bed Occupancy Forecasting")
st.markdown(
    "<h4 style='color:#0068c9;'>Forecast ICU bed usage using historical data to improve resource planning.</h4>",
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    forecast_days = st.slider("Forecast Days", 7, 30, 14, help="Number of days to forecast into the future.")
    max_capacity = st.number_input("ICU Capacity (beds)", min_value=1, value=12, help="Maximum number of ICU beds available.")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload ICU occupancy CSV", type="csv", help="Upload a CSV file with columns: date, icu_occupancy.")
    st.markdown("---")
    st.info("Download the forecast table after running the model.")
    download_placeholder = st.empty()

# --- Main Logic ---
if uploaded_file:
    with st.spinner("Training forecasting model and generating forecast..."):
        df = pd.read_csv(uploaded_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'date': 'ds', 'icu_occupancy': 'y'})

        # Rolling average
        df['7d_avg'] = df['y'].rolling(window=7).mean()

        # Layout: Historical chart and settings side by side
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Historical ICU Bed Occupancy")
            st.line_chart(df.set_index('ds')[['y', '7d_avg']])
        with col2:
            st.markdown("<b>Data Preview</b>", unsafe_allow_html=True)
            st.dataframe(df[['ds', 'y']].tail(10).rename(columns={'ds': 'Date', 'y': 'Occupancy'}), height=250)

        # Prophet Model
        model = Prophet()
        model.fit(df)

        # Forecast
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

    # Forecast Plot
    st.subheader("ICU Occupancy Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#0068c9', width=3)))
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual', line=dict(color='#43aa8b', width=2)))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound', line=dict(dash='dot', color='#f94144')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound', line=dict(dash='dot', color='#f94144')))
    fig.update_layout(title="ICU Occupancy Forecast", xaxis_title="Date", yaxis_title="Occupied Beds", plot_bgcolor="#f5f6fa", paper_bgcolor="#f5f6fa")
    st.plotly_chart(fig, use_container_width=True)

    # Forecast Table and Capacity Warnings in columns
    col3, col4 = st.columns([2, 1])
    with col3:
        forecast_tail = forecast[['ds', 'yhat']].tail(forecast_days)
        forecast_tail['yhat'] = forecast_tail['yhat'].round(1)
        st.subheader("Forecast Table")
        st.dataframe(forecast_tail.rename(columns={'ds': 'Date', 'yhat': 'Forecasted Occupancy'}).style.background_gradient(cmap='Blues'), height=350)
    with col4:
        st.subheader("Capacity Warnings")
        overload = forecast_tail[forecast_tail['yhat'] > max_capacity]
        if not overload.empty:
            st.error(f"{len(overload)} day(s) exceed ICU capacity of {max_capacity} beds.")
            st.dataframe(overload.rename(columns={'ds': 'Date', 'yhat': 'Forecasted Occupancy'}).style.background_gradient(cmap='Reds'), height=200)
        else:
            st.success("All forecasted days are within ICU capacity.")

    # Peak occupancy
    peak_day = forecast_tail.loc[forecast_tail['yhat'].idxmax()]
    st.markdown(f"<b>Peak Predicted Day</b>: <code>{peak_day['ds'].date()}</code> with <b>{peak_day['yhat']} beds</b>.", unsafe_allow_html=True)

    # Trend & Seasonality
    st.subheader("Trend and Seasonality Insights")
    st.plotly_chart(plot_components_plotly(model, forecast), use_container_width=True)

    # Download in sidebar
    csv = forecast_tail.rename(columns={'ds': 'Date', 'yhat': 'Forecasted Occupancy'}).to_csv(index=False)
    with st.sidebar:
        download_placeholder.download_button("Download Forecast CSV", data=csv, file_name="icu_forecast.csv")
else:
    st.info("Please upload a CSV file to begin forecasting.")

# --- Author Footer ---
st.markdown(
    """
    <style>
    .author-footer {
        position: fixed;
        right: 20px;
        bottom: 10px;
        color: #888;
        font-size: 15px;
        z-index: 100;
    }
    </style>
    <div class='author-footer'>Created by <b>Vipul Singh Parmar</b></div>
    """,
    unsafe_allow_html=True,
)
