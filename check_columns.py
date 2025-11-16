import pandas as pd

try:
    df_health = pd.read_csv('air_quality_health_impact_data.csv')
    print("--- Health CSV Columns ---")
    print(df_health.columns.tolist())
    print("\n")

except FileNotFoundError:
    print("Error: 'air_quality_health_impact_data.csv' not found.")
except Exception as e:
    print(f"An error occurred: {e}")

try:
    df_city = pd.read_csv('city_day.csv')
    print("--- City Day CSV Columns ---")
    print(df_city.columns.tolist())

except FileNotFoundError:
    print("Error: 'city_day.csv' not found.")
except Exception as e:
    print(f"An error occurred: {e}")