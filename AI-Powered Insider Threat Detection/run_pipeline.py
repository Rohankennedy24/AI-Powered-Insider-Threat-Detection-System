# run_pipeline.py - Orchestrates Training (main.py logic) and Real-Time Detection

import pandas as pd
import os
import joblib
import time
import random
from datetime import datetime
import numpy as np
from datetime import timedelta

# --- Import Functions from Separate Modules (Must be in src/ folder) ---
# NOTE: You MUST have the following files in your 'src' directory:
# src/data_generator.py
# src/preprocessor.py
# src/model_trainer.py

# In a combined script, we'll redefine the necessary functions here or assume 
# you run the combined script from the main directory where it can access 'src/'.

# --- Redefining Necessary Functions (to make this file self-contained) ---

# --- Functions from src/data_generator.py ---
def generate_employee_roster():
    department_employees = {'IT': 10, 'HR': 5, 'Engineering': 50}
    employees = []
    for dept, count in department_employees.items():
        for i in range(1, count + 1):
            employees.append({'emp_id': f'{dept.upper()}{i:03d}', 'department': dept})
    return pd.DataFrame(employees)

def generate_normal_logs(emp_df):
    START_DATE = datetime(2025, 1, 1)
    SIMULATION_DAYS = 60
    all_logs = []
    base_rules = {
        'IT': {'login_hours': (8, 18), 'files_per_day': (5, 20), 'file_types': ['.conf', '.log', '.sh', '.py']},
        'HR': {'login_hours': (9, 17), 'files_per_day': (1, 10), 'file_types': ['.docx', '.pdf', '.xlsx']},
        'Engineering': {'login_hours': (7, 19), 'files_per_day': (10, 30), 'file_types': ['.cpp', '.java', '.py', '.md']},
    }
    for day in range(SIMULATION_DAYS):
        current_date = START_DATE + timedelta(days=day)
        if current_date.weekday() >= 5: continue
        for _, emp in emp_df.iterrows():
            dept = emp['department']; rules = base_rules[dept]
            login_hour = random.randint(rules['login_hours'][0], rules['login_hours'][0] + 1)
            login_time = current_date + timedelta(hours=login_hour, minutes=random.randint(0, 59))
            all_logs.append({'timestamp': login_time, 'emp_id': emp['emp_id'], 'activity': 'login', 'is_threat': 0, 'data_transferred_mb': np.nan})
            
            num_files = random.randint(rules['files_per_day'][0], rules['files_per_day'][1])
            for _ in range(num_files):
                access_hour = random.randint(login_hour + 1, rules['login_hours'][1])
                access_time = current_date + timedelta(hours=access_hour, minutes=random.randint(0, 59))
                file_ext = random.choice(rules['file_types'])
                all_logs.append({'timestamp': access_time, 'emp_id': emp['emp_id'], 'activity': 'file_access', 'is_threat': 0, 'data_transferred_mb': random.uniform(5.0, 30.0)})
    df_logs = pd.DataFrame(all_logs); df_logs = df_logs.sort_values(by='timestamp').reset_index(drop=True)
    return df_logs

def inject_threats(df_normal_logs):
    all_threat_logs = []
    insider_it_emp = "IT001"; threat_date = datetime(2025, 2, 28) 
    unauthorized_access_time = threat_date + timedelta(hours=23, minutes=30)
    
    # 1. Unusual activity time
    threat_log_1 = {'timestamp': unauthorized_access_time, 'emp_id': insider_it_emp, 'activity': 'file_access', 'data_transferred_mb': 50.0, 'is_threat': 1}
    
    # 2. Exfiltration (Large Data Transfer)
    exfil_time = unauthorized_access_time + timedelta(minutes=15)
    threat_log_2 = {'timestamp': exfil_time, 'emp_id': insider_it_emp, 'activity': 'external_upload', 'data_transferred_mb': 1500.0, 'is_threat': 1}
    
    all_threat_logs.extend([threat_log_1, threat_log_2])
    
    df_threat_logs = pd.DataFrame(all_threat_logs)
    df_final = pd.concat([df_normal_logs, df_threat_logs], ignore_index=True)
    df_final = df_final.sort_values(by='timestamp').reset_index(drop=True)
    df_final['data_transferred_mb'] = df_final['data_transferred_mb'].fillna(0)
    return df_final

# --- Functions from src/preprocessor.py ---
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['data_transferred_mb'] = df['data_transferred_mb'].fillna(0)
    
    # Noise Filtration & Deviation Features
    processed_df = df.groupby(['emp_id', 'date']).agg(
        login_count=('activity', lambda x: (x == 'login').sum()),
        file_access_count=('activity', lambda x: (x == 'file_access').sum()),
        web_browsing_count=('activity', lambda x: (x == 'web_browsing').sum()),
        data_transfer_mb=('data_transferred_mb', 'sum'),
        is_threat=('is_threat', 'max') 
    ).reset_index()

    # Personalized Baselines Feature
    processed_df['day_of_week'] = pd.to_datetime(processed_df['date']).dt.dayofweek
    return processed_df

# --- Functions from src/model_trainer.py ---
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import shap

FEATURES = ['login_count', 'file_access_count', 'web_browsing_count', 'data_transfer_mb', 'day_of_week']

def train_model(processed_df):
    X = processed_df[FEATURES]
    model = IsolationForest(contamination=0.0005, random_state=42, n_jobs=-1)
    model.fit(X)
    return model

def save_model(model, file_path):
    joblib.dump(model, file_path)

def evaluate_model(model, processed_df):
    # Simplified evaluation for conciseness
    X = processed_df[FEATURES]
    y_true = processed_df['is_threat']
    y_pred_labels = model.predict(X)
    y_pred = pd.Series(y_pred_labels).apply(lambda x: 1 if x == -1 else 0)
    
    print(f"Total True Threats: {y_true.sum()}")
    print(f"Total Predicted Threats: {y_pred.sum()}")

def explain_prediction(model, processed_df):
    explainer = shap.TreeExplainer(model)
    threat_rows = processed_df[processed_df['is_threat'] == 1]
    if threat_rows.empty: return
    threat_row = threat_rows.iloc[0]
    threat_data = threat_row[FEATURES]
    shap_values = explainer.shap_values(threat_data)
    
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[0]
        base_value = explainer.expected_value[0]
    else:
        shap_values_to_plot = shap_values
        base_value = explainer.expected_value
        
    # ... inside the explain_prediction function ...
    
    # 4. Generate the SHAP Force Plot and SAVE IT (Fix for no graph)
    plot_filename = "shap_explanation_IT001.png"
    
    print(f"Explaining threat for Employee ID: {threat_row['emp_id']} on {threat_row['date']}")

    # Create the figure object
    plt.figure(figsize=(12, 6)) 

    # Generate the Force Plot (Crucial change: set show=False)
    shap.force_plot(
        base_value, 
        shap_values_to_plot, 
        threat_data, 
        feature_names=FEATURES,
        matplotlib=True,
        show=False, 
    )
    plt.title("SHAP Force Plot: Anomaly Explanation", fontsize=14)

    # -------------------------------------------------------------------
    # --- START: CODE TO FORCE ANOMALY BAR COLOR TO RED ---
    try:
        # Find the primary graphical object (PolyCollection) that forms the colored bar
        for collection in plt.gca().collections:
            # Set the face (fill) and edge color to red
            collection.set_facecolor('red') 
            collection.set_edgecolor('red')
    except Exception as e:
        print(f"Warning: Could not force red color on SHAP plot: {e}")
    # --- END: CODE TO FORCE ANOMALY BAR COLOR TO RED ---
    # -------------------------------------------------------------------

    # *** Save the plot to your main directory ***
    plt.savefig(plot_filename, bbox_inches='tight')

    # *** Save the plot to your main directory ***
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close() 

    print(f"SHAP Plot saved successfully as {plot_filename} in your project directory.")
    
    
    # =========================================================================
    # *** START OF NEW TEXTUAL EXPLANATION CODE BLOCK ***
    # =========================================================================

    print("\n\n--- SHAP Textual Explanation (Why the Threat Happened) ---")
    
    # Analyze the SHAP values (which are in shap_values_to_plot)
    shap_dict = dict(zip(FEATURES, shap_values_to_plot))
    
    # Sort features by absolute SHAP value magnitude to find the most important ones
    sorted_shap = sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)
    
    # Get the feature values for the explanation row
    feature_values = threat_data.to_dict()
    
    print(f"**Anomaly Detected for Employee {threat_row['emp_id']} on {threat_row['date']}**\n")
    print("This activity was flagged as an anomaly (threat) because it significantly deviates from the user's normal baseline (Module 1). The reasons, ranked by impact, are:")
    
    # Print the top 3 most important features
    for rank, (feature, shap_value) in enumerate(sorted_shap[:3]):
        # Determine if the feature is pushing towards Anomaly (negative SHAP) or Normal (positive SHAP)
        direction = "A MAJOR ANOMALY DRIVER" if shap_value < 0 else "A MINOR MITIGATING FACTOR"
        
        # Format output
        print(f"\n{rank+1}. Feature: {feature.upper()}")
        print(f"   - Deviation: Value is {feature_values[feature]:.2f} (Normal baseline is much lower)")
        print(f"   - Impact: {direction}. (SHAP Value: {shap_value:.4f})")
        
        # Provide specific context for the top driver
        if feature == 'data_transfer_mb':
            print("   - Conclusion: This indicates a massive data exfiltration attempt, confirming the threat.")
        elif feature == 'day_of_week':
            print("   - Conclusion: This indicates the activity occurred at an unusual time (off-hours/weekend).")

    print("\n--- End of SHAP Explanation ---")

    # =========================================================================
    # *** END OF NEW TEXTUAL EXPLANATION CODE BLOCK ***
    # =========================================================================
    
    # Remove or comment out the following line if you want to avoid a freezing script
    # plt.show()
    
    # The training pipeline will now continue successfully after printing the report.

# --- Shared Configuration ---
OUTPUT_DIR = "data/synthetic_data"
OUTPUT_FILE = "insider_threat_dataset.csv"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
MODEL_PATH = os.path.join("models", "isolation_forest_model.joblib")


# --- Main Orchestration Logic (Training) ---
def run_training_pipeline():
    print("--- Training Pipeline Started (Modules 1, 2, 3 Setup) ---")

    # 1. Data Generation (Preparation)
    emp_df = generate_employee_roster()
    df_normal_logs = generate_normal_logs(emp_df)
    df_final_logs = inject_threats(df_normal_logs)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    df_final_logs.to_csv(OUTPUT_PATH, index=False)
    
    # 2. Preprocess (Module 1: User Behaviour Monitoring)
    print("\n--- Module 1: Preprocessing & Feature Engineering ---")
    processed_df = preprocess_data(OUTPUT_PATH)
    
    # 3. Train Model (Module 2: Threat Detection - Isolation Forest)
    print("\n--- Module 2: Training Isolation Forest Model ---")
    trained_model = train_model(processed_df)
    save_model(trained_model, MODEL_PATH)

    # 4. Evaluate and Explain (Module 3: Alert & Response - SHAP)
    print("\n--- Module 3: Evaluation & SHAP Explanation ---")
    evaluate_model(trained_model, processed_df)
    explain_prediction(trained_model, processed_df)
    
    print("\n--- Training Pipeline Finished ---")
    
    # Return the trained model for immediate use in real-time mode
    return trained_model

# --- Real-Time Detection Logic ---

def simulate_new_event():
    """Simulates a new user activity event in real-time."""
    activities = ['login', 'file_access', 'web_browsing']
    emp_id = f'EMP{random.randint(1, 100):03d}'
    activity_type = random.choice(activities)
    
    data_mb = random.uniform(0.1, 50.0)
    is_threat = 0

    # 10% chance of a large data transfer (simulated threat)
    if random.random() < 0.1: 
        data_mb = random.uniform(500, 2000) # Anomalously large data transfer
        is_threat = 1

    return {
        'emp_id': emp_id,
        'timestamp': datetime.now(),
        'activity': activity_type,
        'data_transferred_mb': data_mb,
        'is_threat': is_threat
    }
    
def preprocess_realtime_event(event):
    """
    Simulates real-time feature extraction for an individual event
    (simplified version of Module 1 logic for a single data point).
    """
    data = {
        'login_count': [1 if event['activity'] == 'login' else 0],
        'file_access_count': [1 if event['activity'] == 'file_access' else 0],
        'web_browsing_count': [1 if event['activity'] == 'web_browsing' else 0],
        'data_transfer_mb': [event['data_transferred_mb']],
        'day_of_week': [event['timestamp'].weekday()]
    }
    return pd.DataFrame(data)

def run_realtime_detection(model):
    """
    Runs the real-time detection loop (Module 2).
    """
    print("\n--- Starting Real-Time Threat Detection (Module 2) ---")
    print("Press Ctrl+C to stop the process.")

    try:
        while True:
            # Step 1: Simulate a new event arriving
            new_event = simulate_new_event()
            print(f"\nProcessing new event for {new_event['emp_id']} at {new_event['timestamp'].strftime('%H:%M:%S')}...")
            
            # Step 2: Preprocess the event
            processed_event = preprocess_realtime_event(new_event)
            
            # Step 3: Make a prediction using the Isolation Forest
            anomaly_score = model.decision_function(processed_event[FEATURES])
            
            # Step 4: Check if it's a threat (Anomaly score < 0 indicates an anomaly)
            if anomaly_score < 0:
                print("🚨 ANOMALY DETECTED! Potential Insider Threat.")
                print(f"Details: Activity: {new_event['activity']}, Data Transfer: {new_event['data_transferred_mb']:.2f} MB")
                print(f"Anomaly Score: {anomaly_score[0]:.4f}")
                
                # In a real system, you would trigger the SHAP explanation (Module 3) here
            else:
                print("✅ Event is normal.")
                
            # Step 5: Wait for the next event
            time.sleep(1) # Wait 1 second before processing the next event

    except KeyboardInterrupt:
        print("\nReal-Time Detection stopped by user.")
        
# --- Main Execution Block ---
if __name__ == "__main__":
    
    # 1. Run the Training Pipeline (Modules 1, 2, 3 Setup)
    trained_model = run_training_pipeline()
    
    # 2. Run the Real-Time Detection (Module 2)
    # Give the user a moment to review the SHAP plot if it appeared
    print("\n--- Entering Real-Time Mode ---")
    
    try:
        # Try loading the model again in case the user closed the SHAP plot
        # and needs to start the real-time detection later.
        if trained_model is None:
             trained_model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Could not load model for real-time mode: {e}. Exiting.")
        exit()

    run_realtime_detection(trained_model)