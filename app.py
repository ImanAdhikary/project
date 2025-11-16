import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
# New imports for advanced time series analysis
import pmdarima as pm
from pmdarima.arima.utils import ndiffs, nsdiffs # <-- NEW IMPORT
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Air Quality & Health Impact Dashboard",
    page_icon="🌬️",
    layout="wide"
)

# --- 2. CUSTOM CSS INJECTION ---
# [CSS Content Redacted For Brevity]
CUSTOM_CSS = """
<style>
    /* Main app styling */
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
        box-shadow: 2px 0px 10px rgba(0,0,0,0.05);
    }
    
    /* Metric boxes - This is the key "dashboard" look */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: all 0.3s ease-in-out;
    }
    
    [data-testid="stMetric"]:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }

    [data-testid="stMetricLabel"] {
        font-size: 16px;
        color: #555; /* Darker gray for label */
        font-weight: 500;
    }

    [data-testid="stMetricValue"] {
        font-size: 36px;
        font-weight: 700;
        color: #111; /* Black for value */
    }

    /* Tab styling */
    [data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #f0f2f6;
        border-bottom: 3px solid #0068c9;
        color: #0068c9;
        font-weight: 600;
    }
    
    [data-testid="stTabs"] button {
        color: #555;
        font-weight: 500;
        padding: 10px;
    }

    /* Button styling */
    .stButton>button {
        background-color: #0068c9;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%; /* Make buttons full-width in their container */
    }

    .stButton>button:hover {
        background-color: #004a91;
        transform: scale(1.02);
    }
    
    /* Styling for containers (wrapping graphs) */
    .st-emotion-cache-1jicfl2, .st-emotion-cache-0 { 
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    h1, h2, h3 {
        color: #111;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- 3. MODEL & DATA LOADING ---
@st.cache_resource
def load_models():
    """Load all three trained models. Display error in app if models not found."""
    try:
        health_model = joblib.load('health_model.pkl')
    except FileNotFoundError:
        st.error("Fatal Error: 'health_model.pkl' not found. Please run `train_model.py` first.")
        health_model = None
    
    try:
        aqi_bucket_model = joblib.load('aqi_bucket_model.pkl')
    except FileNotFoundError:
        st.error("Fatal Error: 'aqi_bucket_model.pkl' not found. Please run `train_model.py` first.")
        aqi_bucket_model = None
        
    try:
        aqi_score_model = joblib.load('aqi_score_model.pkl')
    except FileNotFoundError:
        st.error("Fatal Error: 'aqi_score_model.pkl' not found. Please run `train_model.py` first.")
        aqi_score_model = None
        
    return health_model, aqi_bucket_model, aqi_score_model

@st.cache_data
def load_data(filepath):
    """Load and preprocess data for visualization."""
    try:
        df = pd.read_csv(filepath)
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df.dropna(subset=['Datetime'], inplace=True)
        df['Year'] = df['Datetime'].dt.year
        df['Month'] = df['Datetime'].dt.month
        df['Month_Name'] = df['Datetime'].dt.month_name()
        
        # Define relevant pollutants
        pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
        
        # Impute 0 for missing pollutant values for plotting
        for col in pollutants:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        return df
    
    except FileNotFoundError:
        st.error(f"Fatal Error: '{filepath}' not found. Cannot load visualization data.")
        return pd.DataFrame()

# Load all assets
health_model, aqi_bucket_model, aqi_score_model = load_models()
df_viz = load_data('city_day.csv')


# --- 4. HELPER FUNCTIONS ---
def generate_conclusion(aqi_bucket, aqi_score, health_preds, inputs):
    """Generates dynamic text conclusions based on model outputs."""
    conclusions = []
    
    # AQI Bucket analysis
    if aqi_bucket == "Severe" or aqi_bucket == "Very Poor":
        conclusions.append(f"**Immediate Health Warning:** The predicted AQI bucket is **{aqi_bucket}** (Score: ~{aqi_score:.0f}). This air quality is hazardous and can cause respiratory illness on prolonged exposure. It seriously impacts those with existing diseases.")
    elif aqi_bucket == "Poor" or aqi_bucket == "Moderate":
        conclusions.append(f"**Health Caution:** Predicted AQI is **{aqi_bucket}** (Score: ~{aqi_score:.0f}). This can cause breathing discomfort to sensitive people and may affect people with lung or heart diseases.")
    else:
        conclusions.append(f"**Good News:** Predicted AQI is **{aqi_bucket}** (Score: ~{aqi_score:.0f}). Air quality is satisfactory and poses minimal health risk.")

    # Health Impact analysis
    conclusions.append(f"The model predicts **{health_preds[0]:.0f} respiratory cases** and **{health_preds[2]:.0f} hospital admissions** under these conditions.")
    
    # Key driver analysis
    input_df = pd.DataFrame([inputs], columns=aqi_score_model_features)
    if input_df['PM2.5'].iloc[0] > 150 or input_df['PM10'].iloc[0] > 300:
        conclusions.append(f"**Key Driver:** The high Particulate Matter (PM2.5: {input_df['PM2.5'].iloc[0]}, PM10: {input_df['PM10'].iloc[0]}) is the primary driver for this poor air quality.")
    elif input_df['O3'].iloc[0] > 100 or input_df['NO2'].iloc[0] > 50:
         conclusions.append(f"**Key Driver:** High levels of gaseous pollutants (O3: {input_df['O3'].iloc[0]}, NO2: {input_df['NO2'].iloc[0]}) are likely contributing to the health risk.")

    return "<ul>" + "".join([f"<li>{c}</li>" for c in conclusions]) + "</ul>"

def get_aqi_color(aqi_bucket):
    """Returns a color based on the AQI bucket for charts."""
    colors = {
        "Good": "#4CAF50",
        "Satisfactory": "#8BC34A",
        "Moderate": "#FFEB3B",
        "Poor": "#FF9800",
        "Very Poor": "#F44336",
        "Severe": "#B71C1C",
        "Default": "#9E9E9E"
    }
    return colors.get(aqi_bucket, "Default")

# Define feature lists (must match training script)
health_model_features = ['PM2_5', 'PM10', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed']
aqi_bucket_model_features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']
aqi_score_model_features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']


# --- 5. SIDEBAR / INPUTS ---
st.sidebar.image("https://placehold.co/300x100/0068c9/FFFFFF?text=Air+Quality+AI", use_column_width=True)
st.sidebar.title("Model Input Panel")
st.sidebar.markdown("Use the sliders to set pollutant and weather values for prediction.")

input_data = {}
st.sidebar.subheader("Pollutant Inputs")
input_data['PM2.5'] = st.sidebar.slider("PM2.5 (μg/m³)", 0.0, 500.0, 150.0, 1.0)
input_data['PM10'] = st.sidebar.slider("PM10 (μg/m³)", 0.0, 700.0, 250.0, 1.0)
input_data['NO2'] = st.sidebar.slider("NO2 (ppb)", 0.0, 200.0, 50.0, 0.5)
input_data['SO2'] = st.sidebar.slider("SO2 (ppb)", 0.0, 200.0, 30.0, 0.5)
input_data['O3'] = st.sidebar.slider("O3 (ppb)", 0.0, 300.0, 40.0, 0.5)
input_data['CO'] = st.sidebar.slider("CO (mg/m³)", 0.0, 20.0, 2.0, 0.1)

st.sidebar.subheader("Weather Inputs (for Health Model)")
input_data['Temperature'] = st.sidebar.slider("Temperature (°C)", -20.0, 50.0, 25.0, 0.5)
input_data['Humidity'] = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0, 1.0)
input_data['WindSpeed'] = st.sidebar.slider("WindSpeed (km/h)", 0.0, 100.0, 10.0, 0.5)


# --- 6. MAIN APPLICATION LAYOUT ---
st.title("Air Quality & Health Impact AI Dashboard 🌬️")
st.markdown("This dashboard uses machine learning to predict air quality and its impact on public health. It also provides deep-dive statistical analysis of historical data.")

# Check if models are loaded before creating tabs
if not all([health_model, aqi_bucket_model, aqi_score_model, not df_viz.empty]):
    st.error("Application cannot start. Please ensure `train_model.py` has been run and all CSV files are present.")
    st.stop()

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction Dashboard", "📈 Historical Analysis", "🔗 Pollutant Relationships", "🔮 Time Series Forecast"])

# --- TAB 1: PREDICTION DASHBOARD ---
with tab1:
    st.header("Real-Time Prediction")
    st.markdown("The models below use your inputs from the sidebar to generate live predictions.")
    
    # Prepare input dataframes for each model
    try:
        # Health model needs 'PM2_5' (underscore)
        input_df_health_dict = {f: input_data[f.replace('_', '.')] for f in health_model_features}
        input_df_health = pd.DataFrame([input_df_health_dict], columns=health_model_features)
        
        # AQI models need 'PM2.5' (dot)
        input_df_aqi = pd.DataFrame([input_data], columns=aqi_bucket_model_features)
        
        # Run all models
        health_prediction = health_model.predict(input_df_health)
        aqi_bucket_prediction = aqi_bucket_model.predict(input_df_aqi)
        aqi_score_prediction = aqi_score_model.predict(input_df_aqi)

        # Get the single prediction values
        health_preds = health_prediction[0]
        aqi_bucket = aqi_bucket_prediction[0]
        aqi_score = aqi_score_prediction[0]
        
        # --- Prediction Metrics Display ---
        st.subheader("Model Outputs")
        cols = st.columns([1.5, 1.5, 1, 1, 1])
        
        # Main AQI Score
        cols[0].metric(label="Predicted AQI Bucket", value=aqi_bucket)
        cols[1].metric(label="Predicted AQI Score", value=f"{aqi_score:.0f}")
        
        # Health Scores
        cols[2].metric(label="Respiratory Cases", value=f"{health_preds[0]:.0f}")
        cols[3].metric(label="Cardio Cases", value=f"{health_preds[1]:.0f}")
        cols[4].metric(label="Hospital Admissions", value=f"{health_preds[2]:.0f}")
        
        st.divider()

        # --- Dynamic Conclusions & Pie Chart ---
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Analysis & Conclusions")
            conclusion_text = generate_conclusion(aqi_bucket, aqi_score, health_preds, input_data)
            st.markdown(conclusion_text, unsafe_allow_html=True)
            
        with col2:
            st.subheader("Input Pollutant Share")
            pie_data = pd.DataFrame({
                'Pollutant': aqi_bucket_model_features,
                'Value': input_df_aqi.iloc[0]
            })
            # Create a Pie chart like the screenshot
            fig_pie = px.pie(pie_data, names='Pollutant', values='Value', 
                             title="Pollutant Input Composition", hole=0.4,
                             color_discrete_sequence=px.colors.sequential.RdBu)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.exception(e)

# --- TAB 2: HISTORICAL ANALYSIS ---
with tab2:
    st.header("Historical Data Explorer")
    st.markdown("Analyze historical trends for different cities and years. Data from `city_day.csv`.")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        city_list = sorted(df_viz['City'].unique().tolist())
        selected_city_hist = st.selectbox("Select City", city_list, key="hist_city")
    with col2:
        year_list = sorted(df_viz['Year'].dropna().unique().astype(int).tolist())
        selected_year_hist = st.selectbox("Select Year", year_list, index=len(year_list)-1, key="hist_year")
        
    # Filter data
    df_hist = df_viz[(df_viz['City'] == selected_city_hist) & (df_viz['Year'] == selected_year_hist)]
    
    if df_hist.empty:
        st.warning("No data available for the selected City and Year.")
    else:
        # --- NEW: Pollutant Distribution Histogram ---
        st.subheader("Pollutant Distribution (Histogram)")
        st.markdown("This shows the frequency of different pollution levels (how many 'low' vs. 'high' days).")
        hist_pollutant = st.selectbox("Select Pollutant for Histogram", aqi_bucket_model_features, key="hist_pollutant")
        fig_hist = px.histogram(df_hist, x=hist_pollutant, nbins=50, title=f"Distribution of {hist_pollutant} Days")
        st.plotly_chart(fig_hist, use_container_width=True)
        # --- END OF NEW ---
        
        st.divider()

        col1, col2 = st.columns(2)
        
        with col1:
            # Bar Chart: Monthly Averages
            st.subheader(f"Monthly Pollutant Averages for {selected_city_hist} ({selected_year_hist})")
            monthly_data = df_hist.groupby('Month_Name')[aqi_bucket_model_features].mean().reset_index()
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            monthly_data = monthly_data.melt(id_vars='Month_Name', var_name='Pollutant', value_name='Average Value')
            monthly_data['Month_Name'] = pd.Categorical(monthly_data['Month_Name'], categories=month_order, ordered=True)
            monthly_data = monthly_data.sort_values('Month_Name')

            fig_bar = px.bar(monthly_data, x='Month_Name', y='Average Value', color='Pollutant',
                             barmode='group', title=f"Average Monthly Pollutant Levels")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            # Donut Chart: AQI Bucket Distribution
            st.subheader(f"AQI Bucket Distribution for {selected_city_hist} ({selected_year_hist})")
            bucket_counts = df_hist['AQI_Bucket'].value_counts().reset_index()
            bucket_counts.columns = ['AQI_Bucket', 'Count']
            
            # Map colors
            bucket_counts['Color'] = bucket_counts['AQI_Bucket'].apply(get_aqi_color)
            
            fig_donut = px.pie(bucket_counts, names='AQI_Bucket', values='Count', 
                               title="Distribution of Air Quality Days", hole=0.5)
            fig_donut.update_traces(textposition='inside', textinfo='percent+label',
                                    marker=dict(colors=bucket_counts['Color'], line=dict(color='#000000', width=1)))
            st.plotly_chart(fig_donut, use_container_width=True)

# --- TAB 3: POLLUTANT RELATIONSHIPS ---
with tab3:
    st.header("Pollutant Relationship Explorer")
    st.markdown("See how different pollutants relate to each other. Select a city to see its specific data.")
    
    col1, col2, col3 = st.columns(3)
    
    # Filters
    city_list_scatter = sorted(df_viz['City'].unique().tolist())
    selected_city_scatter = col1.selectbox("Select City", city_list_scatter, key="scatter_city")
    
    scatter_options = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO', 'AQI']
    
    selected_x = col2.selectbox("Select X-Axis", scatter_options, index=0)
    
    # --- FIX for DuplicateError ---
    dynamic_y_options = [opt for opt in scatter_options if opt != selected_x]
    default_y_index = 0
    if 'PM10' in dynamic_y_options: default_y_index = dynamic_y_options.index('PM10')
    elif 'AQI' in dynamic_y_options: default_y_index = dynamic_y_options.index('AQI')
    selected_y = col3.selectbox("Select Y-Axis", dynamic_y_options, index=default_y_index)
    # --- End of FIX ---
    
    # Filter data
    df_scatter_base = df_viz[df_viz['City'] == selected_city_scatter]
    
    if df_scatter_base.empty:
        st.warning("No data for this city.")
    else:
        # Scatter Plot
        st.subheader(f"{selected_x} vs. {selected_y} for {selected_city_scatter}")
        fig_scatter = px.scatter(df_scatter_base, x=selected_x, y=selected_y, 
                                 opacity=0.5, trendline="ols", trendline_color_override="red",
                                 title=f"Relationship between {selected_x} and {selected_y}")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.divider()

        # --- NEW: Correlation Matrix Heatmap ---
        st.subheader(f"Correlation Matrix (Heatmap) for {selected_city_scatter}")
        st.markdown("This shows the correlation between all pollutants. Values close to 1.0 (dark red) or -1.0 (dark blue) are strong relationships. A high positive correlation between two *features* (like PM2.5 and PM10) is called **multicollinearity**.")
        
        # Calculate correlation
        corr_matrix = df_scatter_base[scatter_options].corr()
        
        # Create heatmap
        fig_heatmap = px.imshow(corr_matrix,
                                text_auto=True, # Show the correlation values
                                aspect="auto",
                                color_continuous_scale='RdBu_r', # Red-Blue scale
                                zmin=-1, zmax=1, # Fix the scale from -1 to 1
                                title=f"Pollutant Correlation Heatmap")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        # --- END OF NEW ---


# --- TAB 4: TIME SERIES ANALYSIS & FORECAST ---
with tab4:
    st.header("Time Series Analysis & Forecast")
    st.markdown("Perform a deep-dive statistical analysis and forecast for a specific pollutant.")
    
    col1, col2 = st.columns(2)
    with col1:
        forecast_city = st.selectbox("Select City", sorted(df_viz['City'].unique().tolist()), key="forecast_city")
    with col2:
        forecast_pollutant = st.selectbox("Select Pollutant", ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO'], key="forecast_pollutant")
    
    # Get the full, clean time series for the selected city
    @st.cache_data
    def get_ts_data(city, pollutant):
        ts_df = df_viz[df_viz['City'] == city][['Datetime', pollutant]]
        ts_df = ts_df.set_index('Datetime')
        ts_df = ts_df.asfreq('D') # Ensure daily frequency
        ts_df = ts_df.interpolate(method='time') # Interpolate missing values
        ts_df = ts_df.dropna()
        return ts_df
    
    ts_data_full = get_ts_data(forecast_city, forecast_pollutant)

    if ts_data_full.empty:
        st.warning("No time series data available for this selection.")
    else:
        # --- 1. Time Series Decomposition ---
        st.subheader(f"1. Time Series Decomposition for {forecast_pollutant}")
        st.markdown("This plot separates the time series into its core components: the overall **Trend**, the **Seasonal** pattern, and the **Residuals** (random noise).")

        # Date Range Slider for Decomposition
        min_date = ts_data_full.index.min().to_pydatetime()
        max_date = ts_data_full.index.max().to_pydatetime()
        
        # FIX for Timestamp mismatch: Convert default value to python datetime
        default_start_ts = max_date - relativedelta(years=2)
        default_start = default_start_ts.to_pydatetime() if isinstance(default_start_ts, pd.Timestamp) else default_start_ts
        
        # Ensure default_start is not before min_date
        if default_start < min_date:
            default_start = min_date

        start_date, end_date = st.slider(
            "Select Date Range to Analyze:",
            min_value=min_date,
            max_value=max_date,
            value=(default_start, max_date),
            format="YYYY-MM-DD",
            key="decomp_slider"
        )
        
        ts_data_decomp = ts_data_full[start_date:end_date]
        
        if len(ts_data_decomp) < (365 * 2):
            st.warning("Please select at least two years of data for an accurate 365-day decomposition.")
        else:
            # Plot Yearly (365-day) Decomposition
            with st.spinner("Calculating 365-day (Yearly) Decomposition..."):
                decomposition_yearly = seasonal_decompose(ts_data_decomp[forecast_pollutant], model='additive', period=365)
                
                # --- FIX: Capture the figure directly from the .plot() method ---
                fig_decomp_yearly = decomposition_yearly.plot()
                fig_decomp_yearly.set_size_inches(10, 8) # Resize the figure after creating it
                # --- End of FIX ---
                
                plt.suptitle("Yearly (365-day) Decomposition", y=1.02)
                st.pyplot(fig_decomp_yearly)
        
        # Plot Weekly (7-day) Decomposition
        if len(ts_data_decomp) < 14:
            st.warning("Please select at least two weeks of data for 7-day decomposition.")
        else:
            with st.spinner("Calculating 7-day (Weekly) Decomposition..."):
                decomposition_weekly = seasonal_decompose(ts_data_decomp[forecast_pollutant], model='additive', period=7)
                
                # --- FIX: Capture the figure directly from the .plot() method ---
                fig_decomp_weekly = decomposition_weekly.plot()
                fig_decomp_weekly.set_size_inches(10, 8) # Resize the figure after creating it
                # --- End of FIX ---
                
                plt.suptitle("Weekly (7-day) Decomposition", y=1.02)
                st.pyplot(fig_decomp_weekly)

        
        st.divider()
        st.subheader("2. ARIMA Model & Forecast")
        
        if st.button(f"Run ARIMA Forecast for {forecast_pollutant} in {forecast_city}"):
            
            with st.spinner(f"Finding best ARIMA model (m=52, yearly seasonality on weekly data)... This may take 30-60 seconds."):
                
                # --- Step 2: Stationarity & Pre-Analysis (on daily data) ---
                st.write("**Stationarity Analysis (ACF/PACF on Daily Data)**")
                st.markdown("First, we check if the *daily* data is 'stationary' (d-value) and look for auto-correlations (p and q values).")
                
                # ADF Test
                adf_test = adfuller(ts_data_full[forecast_pollutant].dropna())
                st.text(f"ADF Statistic: {adf_test[0]:.2f} | p-value: {adf_test[1]:.3f}")
                if adf_test[1] > 0.05:
                    st.warning("Data is non-stationary (p > 0.05). We will use differencing (d=1).")
                    ts_diff = ts_data_full[forecast_pollutant].dropna().diff().dropna()
                else:
                    st.success("Data is stationary (p <= 0.05).")
                    ts_diff = ts_data_full[forecast_pollutant].dropna()
                
                # Plot ACF/PACF
                fig_acf_pacf, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                plot_acf(ts_diff, ax=ax1, lags=40, title="Autocorrelation (ACF)")
                plot_pacf(ts_diff, ax=ax2, lags=40, title="Partial Autocorrelation (PACF)")
                st.pyplot(fig_acf_pacf)
                
                # --- FIX: Resample to Weekly Data for a better, smoother forecast ---
                st.write("--- Resampling to Weekly Data for a Clearer Forecast ---")
                st.write("Daily data is very 'noisy'. Resampling to weekly averages helps the model find the true seasonal pattern (m=52).")
                ts_data_weekly = ts_data_full[forecast_pollutant].resample('W').mean()
                ts_data_weekly = ts_data_weekly.interpolate(method='time').dropna()

                if len(ts_data_weekly) < 104: # Need 2 full seasonal cycles (2 years)
                    st.error(f"Not enough weekly data (minimum 104 weeks required) to find a yearly seasonal pattern. Found only {len(ts_data_weekly)}. Forecast may be flat.")
                    st.stop()
                # --- End of FIX ---

                # --- Step 3: Auto-ARIMA on WEEKLY data ---
                auto_model = pm.auto_arima(
                    ts_data_weekly, # <-- CHANGED
                    start_p=1, start_q=1,
                    test='adf', max_p=3, max_q=3,
                    m=52, # <-- CHANGED: 52 weeks in a year (yearly seasonality)
                    start_P=0, seasonal=True,
                    d=None, D=None, 
                    trace=False, error_action='ignore',  
                    suppress_warnings=True, stepwise=False # <-- FIX: stepwise=False for a full search
                )
                
                # Fit the final model on WEEKLY data
                final_model = ARIMA(ts_data_weekly, # <-- CHANGED
                                    order=auto_model.order, 
                                    seasonal_order=auto_model.seasonal_order)
                final_model_fit = final_model.fit()

                # --- Step 4: Model Summary ---
                st.write(f"**Best Model Found (on Weekly Data):** `ARIMA {auto_model.order}` with `Seasonal {auto_model.seasonal_order}` (m=52)")
                st.text("--- Full Model Summary ---")
                st.text(final_model_fit.summary().as_text())

                # --- Step 5: Model Diagnostics ---
                st.write("--- Full Model Diagnostics ---")
                st.markdown("These plots check the 'health' of the model. **Goal:** The residuals should look like random noise (no patterns). The histogram should be 'Normal' (bell-shaped).")
                fig_diag = plt.figure(figsize=(10, 8))
                final_model_fit.plot_diagnostics(fig=fig_diag)
                st.pyplot(fig_diag)

                # --- Step 6: Forecast Plot ---
                st.write("**1-Year (52-Week) Forecast Plot**") # <-- CHANGED
                # Forecast 52 weeks (1 year) into the future
                forecast_result = final_model_fit.get_forecast(steps=52) # <-- THIS IS THE CHANGE
                forecast = forecast_result.predicted_mean
                conf_int = forecast_result.conf_int(alpha=0.05)
                
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(
                    # Plot last 2 years (104 weeks) of weekly data
                    x=ts_data_weekly.index[-104:], # <-- CHANGED
                    y=ts_data_weekly.iloc[-104:],    # <-- CHANGED
                    mode='lines', name='Actual Data (Weekly Avg, Last 2 Years)'
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=forecast.index, y=forecast,
                    mode='lines', name='Forecasted Values', line=dict(color='green', dash='dash')
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=conf_int.index, y=conf_int.iloc[:, 0],
                    mode='lines', name='95% Conf. Interval (Lower)', line=dict(color='green', width=0)
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=conf_int.index, y=conf_int.iloc[:, 1],
                    mode='lines', name='95% Conf. Interval (Upper)', line=dict(color='green', width=0),
                    fill='tonexty', fillcolor='rgba(0,128,0,0.1)'
                ))
                
                fig_forecast.update_layout(
                    title=f"1-Year (52-Week) Forecast for {forecast_pollutant} in {forecast_city} (Weekly Avg)", # <-- CHANGED
                    xaxis_title="Date", yaxis_title=f"{forecast_pollutant} Level (Weekly Avg)"
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
            
            st.success("Full analysis complete.")
