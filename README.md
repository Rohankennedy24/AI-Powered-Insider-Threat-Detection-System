# AI-Powered-Insider-Threat-Detection-System
An AI-driven system that uses the Isolation Forest model to detect subtle, unknown anomalies in user activity logs. It utilizes User Behavior Monitoring for features and SHAP (Explainable AI) to instantly provide clear, auditable reasons for every threat alert, overcoming the limitations of static, rule-based security.

*Project Highlights*
-->Core Model: Isolation Forest (Unsupervised Anomaly Detection).
-->Key Innovation: Uses Personalized Baselines to detect deviations unique to each employee's history.
-->Transparency: Implements SHAP (Explainable AI) to provide clear, auditable reasons for every threat alert, solving the "black box" problem.

*Quick Start (How to Run)*
This project runs entirely using Python.

1. Prerequisites
You need Python 3.8+ and the following libraries:
pip install pandas scikit-learn joblib shap matplotlib numpy

2. Execution
Run the main orchestration script from the project root directory:
python run_pipeline.py

*This single command will execute the entire process:*
-->Data Generation (creates synthetic logs with injected threats).
-->Model Training (trains the Isolation Forest model).
-->SHAP Explanation (generates shap_explanation_IT001.png for a detected anomaly).
-->Real-Time Simulation (enters a loop to demonstrate live detection)

*Project Architecture (The ML Pipeline)*
The system is built as a three-module pipeline managed by run_pipeline.py.

-->Module 1: User Behaviour Monitoring (src/preprocessor.py)
Purpose: Transforms raw logs into actionable features.
Concept: Implements Noise Filtering (aggregates data by user/day) and creates Deviation Features (e.g., data_transfer_mb, login_count) and Personalized Baselines (day_of_week).

-->Module 2: Threat Detection Engine (src/model_trainer.py)
Purpose: Identifies anomalies in real-time.
Algorithm: Isolation Forest. This model is trained only on "normal" data to actively isolate outliers.
Output: Generates a Negative Anomaly Score for threats (short path length in the model's trees) and a positive score for normal activity.

-->Module 3: Alert & Response (Explainable AI) (src/model_trainer.py)
Purpose: Provides context and justification for alerts.
Tool: SHAP (Shapley Additive Explanations).
Output: Produces a visual Force Plot and a structured textual report detailing exactly which features (data_transfer_mb high, day_of_week unusual) contributed to the anomaly score.
