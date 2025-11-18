
# 🚆 Model Development – Charlie MBTA Delay Prediction  
Full End-to-End Machine Learning Development Pipeline

This module contains the entire machine learning lifecycle for the **MBTA Delay Prediction System**, including:

- Model Training  
- Hyperparameter Tuning  
- Model Selection  
- SHAP + LIME Explainability  
- Bias & Fairness Analysis  
- Drift Monitoring  
- MLflow Tracking  
- Deployment to GCP Artifact Registry  
- CI/CD Automation  

---

### 📁 1. Folder Structure

```bash
Model_Development/
│
├── src/
│   ├── model_train.py
│   ├── model_tuning.py
│   ├── model_select.py
│   ├── bias_analysis.py
│   ├── explainability.py
│   ├── monitor_drift.py
│   ├── register_model.py
│   ├── gcp_registry.py
│   └── utils/
│
├── models/
│   ├── final_model.joblib
│   ├── logreg_tuned.joblib
│   ├── model_lgbm.joblib
│
├── reports/
│   ├── model_comparison.json
│   ├── model_comparison.png
│   ├── shap_importance.csv
│   ├── shap_summary.png
│   ├── lime_explanation.html
│   ├── fairness_by_direction.png
│   ├── drift_report.json
│   └── drift_report.html
│
└── screenshots/
```

⸻

### 🖼 2. Screenshots Included
``` bash
Model_Development/screenshots/
│
├── model_train_output.png
├── model_tuning_output.png
├── model_fairness.png
├── model_explainability.png
├── drift_monitoring_output.png
├── mlflow_home.png
├── mlflow_all_runs.png
├── mlflow_drift_run.png
└── mlflow_registry.png

```
⸻

### 📦 3. Data Loading

All ML scripts automatically load DVC-tracked processed data:
``` bash
Data_Pipeline/data/processed/predictions.csv
Data_Pipeline/data/processed/vehicles.csv
Data_Pipeline/data/processed/alerts.csv

Loader script:

src/data_loader.py
``` 

⸻

### 🤖 4. Model Training

Models trained:
	•	Logistic Regression
	•	LightGBM (Final Winner)

Command:
``` bash
python -m Model_Development.ml_src.model_train
``` 
Logged to MLflow:

accuracy  
precision  
recall  
f1  
roc_auc  


⸻

### 🔧 5. Hyperparameter Tuning (SMOTE + GridSearch)

python -m Model_Development.ml_src.model_tuning

Outputs:
``` bash
models/logreg_tuned.joblib
reports/model_comparison.json
reports/model_comparison.png

``` 
⸻

### 🏆 6. Model Selection

Compares:

accuracy  
f1  
roc_auc  

Run:
``` bash
python -m Model_Development.ml_src.model_select

Final model saved as:

models/final_model.joblib

``` 
⸻

### ✔ 7. Model Validation

Validation includes:

Hold-out split  
5-fold CV  
AUC-ROC  
Confusion matrix  
Precision/Recall  


⸻

### ⚖ 8. Bias & Fairness Analysis (Fairlearn)

Run:
``` bash
python -m Model_Development.ml_src.model_fairness

Outputs:

reports/fairness_by_direction.png
reports/fairness_metrics.csv

``` 
⸻

### 🧠 9. Explainability (SHAP + LIME)

Run:
``` bash
python -m Model_Development.ml_src.model_explain

Outputs:

reports/shap_summary.png
reports/shap_importance.csv
reports/lime_explanation.html

``` 
⸻

### 📉 10. Drift Monitoring

Run:
``` bash
python -m Model_Development.ml_src.monitor_drift

Checks:

Feature drift  
Target drift  
Population Stability Index (PSI)  
Distribution shifts  

Outputs:

reports/drift_report.json
reports/drift_report.html
```

⸻

### ☁ 11. Deployment – GCP Artifact Registry

Run:
``` bash
python -m Model_Development.ml_src.gcp_registry

Uploads:

models/final_model.joblib
models/model_metadata.json

Destination:

artifactregistry.googleapis.com/projects/charlie-478223/...

```
⸻

### 🔁 12. CI/CD (GitHub Actions)

Pipeline file:
``` bash
.github/workflows/mlops_pipeline.yml
``` 
Automated steps:

✔ Train model
✔ Tune model
✔ Bias analysis
✔ Explainability
✔ Drift monitoring
✔ Upload artifacts
✔ Register model
✔ Push to GCP


⸻

### 🧪 13. Run Everything Locally
``` bash
# Install dependencies
pip install -r requirements.txt

# Train baseline model
python -m Model_Development.ml_src.model_train

# Run tuning
python -m Model_Development.ml_src.model_tuning

# Select best model
python -m Model_Development.ml_src.model_select

# Bias analysis
python -m Model_Development.ml_src.model_fairness

# Explainability
python -m Model_Development.ml_src.model_explain

# Drift monitoring
python -m Model_Development.ml_src.monitor_drift

# Push to GCP
python -m Model_Development.ml_src.register_model
``` 



git commit -m "Updated Model Development README with screenshots"
git push origin main


⸻
