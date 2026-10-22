import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run_logistics_analysis():
    print("Initializing Logistics Data Analysis Pipeline...")

    # Mocking dataframe structure for demonstration
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'order_date': pd.date_range(start='2026-10-01', periods=n_samples, freq='H'),
        'destination_lat': np.random.uniform(22.5, 23.5, n_samples),
        'destination_lon': np.random.uniform(77.0, 78.0, n_samples),
        'distance_km': np.random.uniform(2.0, 25.0, n_samples),
        'package_weight_kg': np.random.uniform(0.5, 15.0, n_samples),
        'traffic_index': np.random.uniform(1.0, 5.0, n_samples),
        'driver_experience_yrs': np.random.randint(1, 10, n_samples),
        'delivery_time_hours': np.random.uniform(0.5, 3.5, n_samples)
    })

    # Data Preprocessing
    print("\n--- Phase 2: Data Preprocessing ---")
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['delivery_time_hours'].fillna(df['delivery_time_hours'].median(), inplace=True)
    print("Missing values handled and datetime formatted successfully.")

    # K-Means Clustering for Delivery Zone Segmentation
    print("\n--- Phase 3: Spatial Clustering (K-Means) ---")
    coordinates = df[['destination_lat', 'destination_lon']]
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['delivery_cluster'] = kmeans.fit_predict(coordinates)
    print("Cluster Centers (Hub Coordinates Identified):")
    print(kmeans.cluster_centers_)

    # Predictive Modeling for Delivery Time Estimation
    print("\n--- Phase 4: Predictive Modeling (Random Forest) ---")
    X = df[['distance_km', 'package_weight_kg', 'traffic_index', 'driver_experience_yrs']]
    y = df['delivery_time_hours']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    predictions = rf_model.predict(X_test)
    print("Model Evaluation Metrics:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.4f}")
    print(f"R2 Score: {r2_score(y_test, predictions):.4f}")

if __name__ == "__main__":
    run_logistics_analysis()
  
