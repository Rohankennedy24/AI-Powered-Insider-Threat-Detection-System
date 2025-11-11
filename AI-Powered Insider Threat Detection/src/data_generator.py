import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- Configuration ---
# Set the time frame for the simulation
START_DATE = datetime(2025, 1, 1)
SIMULATION_DAYS = 60
TOTAL_LOGS_PER_EMPLOYEE_DAY = 10 # Average number of activities per employee per day

def generate_employee_roster():
    """Generates a DataFrame of employees with their departments."""
    department_employees = {
        'IT': 10,
        'HR': 5,
        'Engineering': 50,
        'Marketing': 15,
        'Sales': 20
    }

    employees = []
    for dept, count in department_employees.items():
        for i in range(1, count + 1):
            emp_id = f'{dept.upper()}{i:03d}'
            employees.append({'emp_id': emp_id, 'department': dept})

    return pd.DataFrame(employees)

def generate_normal_logs(emp_df):
    """Simulates normal daily activity for all employees based on their department rules."""
    all_logs = []
    
    # Department-specific normal behavior rules
    base_rules = {
        'IT': {'login_hours': (8, 18), 'files_per_day': (5, 20), 'file_types': ['.conf', '.log', '.sh', '.py']},
        'HR': {'login_hours': (9, 17), 'files_per_day': (1, 10), 'file_types': ['.docx', '.pdf', '.xlsx']},
        'Engineering': {'login_hours': (7, 19), 'files_per_day': (10, 30), 'file_types': ['.cpp', '.java', '.py', '.md']},
        'Marketing': {'login_hours': (9, 18), 'files_per_day': (2, 12), 'file_types': ['.pptx', '.jpg', '.ai']},
        'Sales': {'login_hours': (8, 17), 'files_per_day': (3, 15), 'file_types': ['.xlsx', '.docx', '.pdf']}
    }

    for day in range(SIMULATION_DAYS):
        current_date = START_DATE + timedelta(days=day)
        
        # Skip weekends (Saturday=5, Sunday=6)
        if current_date.weekday() >= 5:
            continue
        
        for index, emp in emp_df.iterrows():
            dept = emp['department']
            rules = base_rules[dept]
            
            # 1. Login Event
            login_hour = random.randint(rules['login_hours'][0], rules['login_hours'][0] + 1)
            login_time = current_date + timedelta(hours=login_hour, minutes=random.randint(0, 59))
            all_logs.append({
                'timestamp': login_time,
                'emp_id': emp['emp_id'],
                'activity': 'login',
                'details': 'Success',
                'is_threat': 0,
                'data_transferred_mb': np.nan
            })

            # 2. File Access Events
            num_files = random.randint(rules['files_per_day'][0], rules['files_per_day'][1])
            for _ in range(num_files):
                # File access occurs between login hour and end of shift
                access_hour = random.randint(login_hour + 1, rules['login_hours'][1])
                access_time = current_date + timedelta(hours=access_hour, minutes=random.randint(0, 59))
                file_ext = random.choice(rules['file_types'])
                
                all_logs.append({
                    'timestamp': access_time,
                    'emp_id': emp['emp_id'],
                    'activity': 'file_access',
                    'details': f'Accessed file.ext{file_ext}',
                    'is_threat': 0,
                    # Normal data transfers are usually small
                    'data_transferred_mb': random.uniform(5.0, 30.0)
                })

            # 3. Web Browsing Events (general background activity)
            num_browsing = random.randint(1, 5)
            for _ in range(num_browsing):
                 browsing_hour = random.randint(login_hour + 1, rules['login_hours'][1] - 1)
                 browsing_time = current_date + timedelta(hours=browsing_hour, minutes=random.randint(0, 59))
                 all_logs.append({
                    'timestamp': browsing_time,
                    'emp_id': emp['emp_id'],
                    'activity': 'web_browsing',
                    'details': 'Visited work-related site.',
                    'is_threat': 0,
                    'data_transferred_mb': np.nan
                })
    
    df_logs = pd.DataFrame(all_logs)
    df_logs = df_logs.sort_values(by='timestamp').reset_index(drop=True)
    return df_logs


def inject_threats(df_normal_logs):
    """
    Injects specific insider threat scenarios into the normal log data.
    These are the 'anomalies' the Isolation Forest must detect.
    """
    all_threat_logs = []

    # --- Scenario A: Malicious IT Employee (Data Exfiltration) ---
    insider_it_emp = "IT001"
    # Choose a date outside the normal range, or a highly unusual time
    threat_date = datetime(2025, 2, 28) 
    
    # 1. Unusual activity time (after hours)
    unauthorized_access_time = threat_date + timedelta(hours=23, minutes=30)
    threat_log_1 = {
        'timestamp': unauthorized_access_time,
        'emp_id': insider_it_emp,
        'activity': 'file_access',
        'details': 'Accessed confidential HR docs.',
        'data_transferred_mb': 50.0, # Normal data size, but unusual context
        'is_threat': 1
    }

    # 2. Exfiltration (Large Data Transfer)
    exfil_time = unauthorized_access_time + timedelta(minutes=15)
    threat_log_2 = {
        'timestamp': exfil_time,
        'emp_id': insider_it_emp,
        'activity': 'external_upload',
        'details': 'Uploaded large file to personal cloud.',
        # This large size is the primary feature that will flag the anomaly
        'data_transferred_mb': 1500.0, 
        'is_threat': 1
    }
    
    all_threat_logs.extend([threat_log_1, threat_log_2])

    # --- Scenario B: Negligent HR Employee (Phishing Click) ---
    insider_hr_emp = "HR001"
    threat_date_hr = datetime(2025, 2, 20)
    
    # 3. Phishing link click resulting in unusual web activity
    phish_click_time = threat_date_hr + timedelta(hours=10, minutes=45)
    threat_log_3 = {
        'timestamp': phish_click_time,
        'emp_id': insider_hr_emp,
        'activity': 'web_browsing',
        'details': 'Visited suspicious external domain.',
        'data_transferred_mb': np.nan,
        'is_threat': 1
    }
    
    all_threat_logs.append(threat_log_3)

    # Combine normal logs with threats
    df_threat_logs = pd.DataFrame(all_threat_logs)
    df_final = pd.concat([df_normal_logs, df_threat_logs], ignore_index=True)
    
    # Sort the final dataset by timestamp
    df_final = df_final.sort_values(by='timestamp').reset_index(drop=True)
    
    # Replace NaN data transfers with 0 for aggregation step in preprocessor
    df_final['data_transferred_mb'] = df_final['data_transferred_mb'].fillna(0)
    
    return df_final