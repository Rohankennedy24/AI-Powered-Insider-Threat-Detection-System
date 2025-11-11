# src/model_trainer.py

import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt

# Define the features derived from preprocessor.py
FEATURES = [
    'login_count',
    'file_access_count',
    'web_browsing_count',
    'data_transfer_mb',
    'day_of_week'
]

def train_model(processed_df):
    """
    Trains the Isolation Forest model (Module 2).
    
    Args:
        processed_df (pd.DataFrame): The preprocessed DataFrame with Deviation Features.
        
    Returns:
        The trained IsolationForest model.
    """
    print("Training the Isolation Forest model...")
    
    X = processed_df[FEATURES]
    
    # Isolation Forest for Anomaly Detection (Threat Detection)
    # The contamination parameter estimates the proportion of anomalies in the dataset.
    # Setting it to 0.0005 tells the model to expect a very small percentage of anomalies.
    model = IsolationForest(
        contamination=0.0005, 
        random_state=42,
        n_jobs=-1 # Use all processors for faster training
    )
    
    model.fit(X)
    
    print("Model training complete.")
    return model

def save_model(model, file_path):
    """Saves the trained model to a file."""
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def evaluate_model(model, processed_df):
    """
    Evaluates the model's performance on the training data.
    """
    print("\n--- Model Evaluation ---")
    X = processed_df[FEATURES]
    y_true = processed_df['is_threat']
    
    # Predict the anomaly labels: -1 for anomaly (threat), 1 for inlier (normal)
    y_pred_labels = model.predict(X)
    
    # Convert Isolation Forest labels (-1, 1) to binary threat labels (1, 0)
    # Threat: 1 (original) -> -1 (predicted)
    # Normal: 0 (original) -> 1 (predicted)
    # We map y_pred_labels == -1 to 1 (Threat) and y_pred_labels == 1 to 0 (Normal)
    y_pred = pd.Series(y_pred_labels).apply(lambda x: 1 if x == -1 else 0)

    print("Classification Report (Anomaly: 1, Normal: 0):\n")
    print(classification_report(y_true, y_pred, target_names=['Normal (0)', 'Threat (1)']))


def explain_prediction(model, processed_df):
    """
    Uses SHAP (Module 3) to explain why a specific log was flagged as an anomaly.
    This generates the visual alert for the Alert & Response team.
    
    Args:
        model (IsolationForest): The trained model.
        processed_df (pd.DataFrame): The data containing the anomalies.
    """
    print("\n--- Alert & Response: Explaining Anomalous Behavior (SHAP) ---")
    
    # 1. Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # 2. Identify the first injected threat for explanation
    threat_rows = processed_df[processed_df['is_threat'] == 1]
    if threat_rows.empty:
        print("No threats found in the dataset to explain.")
        return
        
    threat_row = threat_rows.iloc[0]
    threat_data = threat_row[FEATURES] # The Deviation Features for the anomaly
    
    # 3. Calculate SHAP values
    shap_values = explainer.shap_values(threat_data)
    
    # Handle the structure of IsolationForest's output for SHAP
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[0]
        base_value = explainer.expected_value[0]
    else:
        shap_values_to_plot = shap_values
        base_value = explainer.expected_value
        
    # 4. Generate the SHAP Force Plot (The Visual Alert)
    print(f"Explaining threat for Employee ID: {threat_row['emp_id']} on {threat_row['date']}")
    
    plt.figure(figsize=(10, 6))
    shap.force_plot(
        base_value, 
        shap_values_to_plot, 
        threat_data, 
        feature_names=FEATURES,
        matplotlib=True,
        show=True # Displays the plot in a new window
    )
    plt.title("SHAP Force Plot: Anomaly Explanation ")