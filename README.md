# ICU Bed Occupancy Forecasting App

A web application to forecast ICU bed usage using historical data, built with Streamlit and Prophet. This tool helps hospitals and healthcare planners anticipate ICU demand and manage resources more effectively.

## Features
- Upload your own ICU occupancy CSV data
- Visualize historical occupancy and rolling averages
- Forecast future ICU bed usage with Prophet
- Interactive charts and tables
- Capacity warnings and peak day prediction
- Downloadable forecast results
- Clean, modern UI

## Installation
1. Clone this repository or download the source code.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run icu_forecast_app.py
   ```
2. Open the app in your browser (usually at http://localhost:8501).
3. Use the sidebar to upload your ICU occupancy CSV file (with columns: `date`, `icu_occupancy`).
4. Adjust forecast days and ICU capacity as needed.
5. View forecasts, warnings, and download results.

## Data Format
Your CSV should have at least these columns:
- `date` (YYYY-MM-DD)
- `icu_occupancy` (integer)

Example:
```csv
date,icu_occupancy
2024-01-01,8
2024-01-02,10
...
```

## Author
Created by Vipul Singh Parmar
