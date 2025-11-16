import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

print("Starting model training...")

# --- Model 1: Health Impact Prediction (Regression) ---
try:
    df_health = pd.read_csv('air_quality_health_impact_data.csv')

    # Features: All pollutants + weather data
    features_health = ['PM2_5', 'PM10', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed']
    # Targets: All health outcomes
    targets_health = ['RespiratoryCases', 'CardiovascularCases', 'HospitalAdmissions', 'HealthImpactScore']
    
    # We need to drop rows where the *targets* are missing
    df_health = df_health.dropna(subset=targets_health)
    
    X_health = df_health[features_health]
    Y_health = df_health[targets_health]

    # Create a pipeline: 
    # 1. Impute missing (NaN) pollutant/weather values with the mean
    # 2. Scale all features
    # 3. Run the regression model
    health_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', MultiOutputRegressor(LinearRegression()))
    ])

    health_pipeline.fit(X_health, Y_health)
    joblib.dump(health_pipeline, 'health_model.pkl')
    print("1. Health prediction model (Regression) trained and saved as 'health_model.pkl'")

except FileNotFoundError:
    print("Error: 'air_quality_health_impact_data.csv' not found.")
except Exception as e:
    print(f"An error occurred during health model training: {e}")


# --- Model 2: AQI Bucket Prediction (Classification) ---
try:
    df_city = pd.read_csv('city_day.csv') 

    features_aqi = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']
    target_aqi_bucket = 'AQI_Bucket'
    
    # Drop rows where the target is missing
    df_city_bucket = df_city.dropna(subset=[target_aqi_bucket] + features_aqi)

    X_aqi_bucket = df_city_bucket[features_aqi]
    Y_aqi_bucket = df_city_bucket[target_aqi_bucket]

    # Create a pipeline
    aqi_bucket_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)) # n_jobs=-1 uses all cores
    ])

    aqi_bucket_pipeline.fit(X_aqi_bucket, Y_aqi_bucket)
    joblib.dump(aqi_bucket_pipeline, 'aqi_bucket_model.pkl') # Corrected name
    print("2. AQI Bucket prediction model (Classification) trained and saved as 'aqi_bucket_model.pkl'")

except FileNotFoundError:
    print("Error: 'city_day.csv' not found.")
except Exception as e:
    print(f"An error occurred during AQI Bucket model training: {e}")


# --- Model 3: AQI Score Prediction (Regression) ---
try:
    # We can reuse the df_city data loaded for Model 2
    if 'df_city' not in locals():
        df_city = pd.read_csv('city_day.csv') 

    features_aqi = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']
    target_aqi_score = 'AQI'
    
    # Drop rows where the target is missing
    df_city_score = df_city.dropna(subset=[target_aqi_score] + features_aqi)

    X_aqi_score = df_city_score[features_aqi]
    Y_aqi_score = df_city_score[target_aqi_score]

    # Create a pipeline
    aqi_score_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    aqi_score_pipeline.fit(X_aqi_score, Y_aqi_score)
    joblib.dump(aqi_score_pipeline, 'aqi_score_model.pkl') # New model file
    print("3. AQI Score prediction model (Regression) trained and saved as 'aqi_score_model.pkl'")

except FileNotFoundError:
    print("Error: 'city_day.csv' not found.")
except Exception as e:
    print(f"An error occurred during AQI Score model training: {e}")

print("\nTraining complete. All three models are saved.")