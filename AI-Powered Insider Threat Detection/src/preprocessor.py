# src/preprocessor.py

import pandas as pd
import os

def preprocess_data(file_path):
    """
    Loads raw log data, performs aggregation for noise filtration, and 
    engineers deviation features and baseline features for modeling.
    
    Args:
        file_path (str): The path to the raw log data CSV file.
    
    Returns:
        pd.DataFrame: A DataFrame with engineered features ready for modeling.
    """
    print("Loading raw data...")
    df = pd.read_csv(file_path)

    # 1. Prepare Data
    # Convert timestamp to datetime objects and create a date column
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # Fill NaN for data_transferred_mb (e.g., login events have no transfer data)
    df['data_transferred_mb'] = df['data_transferred_mb'].fillna(0)

    # 2. Noise Filtration & Deviation Features
    # Grouping by user (emp_id) and day (date) to summarize behavior.
    # This aggregation step acts as the primary NOISE FILTRATION.
    processed_df = df.groupby(['emp_id', 'date']).agg(
        # Deviation Features: Counting the frequency of key activities
        login_count=('activity', lambda x: (x == 'login').sum()),
        file_access_count=('activity', lambda x: (x == 'file_access').sum()),
        web_browsing_count=('activity', lambda x: (x == 'web_browsing').sum()),
        # Deviation Feature: Summing up total data transferred (key anomaly indicator)
        data_transfer_mb=('data_transferred_mb', 'sum'),
        # Carry over the threat label (1 if ANY log on that day was a threat, 0 otherwise)
        is_threat=('is_threat', 'max') 
    ).reset_index()

    # 3. Personalized Baselines Feature
    # Day of week is crucial for time-based personalized baselines 
    # (e.g., activity differs on Mondays vs. Fridays)
    processed_df['day_of_week'] = pd.to_datetime(processed_df['date']).dt.dayofweek

    print(f"Feature engineering complete. Dataset shape: {processed_df.shape}")
    return processed_df