Project Overview

This project implements a fully automated end-to-end MLOps pipeline to predict train delay probabilities for Boston’s MBTA transit system. The system ingests real-time transit data, processes it, trains multiple ML models, performs bias and drift checks, registers the final model, deploys it as an API using Google Cloud Run, and exposes predictions to users via a Streamlit web application.

This repository demonstrates industry-level MLOps including:
	•	Automated Data Pipelines (Airflow + DVC)
	•	Model Training, Tuning, and Versioning
	•	Bias & Drift Monitoring
	•	CI/CD Pipeline with GitHub Actions
	•	Deployment to Google Cloud Run
	•	Streamlit Frontend
	•	Artifact Registry Model Storage
	•	Experiment Tracking (MLflow)

⸻

 System Architecture

     MBTA API → Airflow Data Pipeline → DVC Versioning → Feature Store
                        ↓
              Model Training (MLflow)
                        ↓
      Bias + Drift Analysis (Fairlearn, Evidently)
                        ↓
       Best Model Selection + Registry (GCP Artifact Registry)
                        ↓
        FastAPI Model Serving (Cloud Run Deployment)
                        ↓
              Streamlit UI for Real-Time Queries


⸻

 Key Components

1. Data Pipeline (Apache Airflow)

Includes 4 DAGs:

DAG	Purpose
data_collection_dag	Fetch live MBTA data periodically
data_processing_dag	Preprocess & structure the data
data_quality_dag	Apply schema checks, anomaly detection
mbta_final_data_pipeline	Full end-to-end execution & DVC integration


⸻

2. Data Versioning (DVC)

All datasets and feature outputs are tracked via DVC for:

✔ reproducibility
✔ data lineage
✔ connection with CI/CD

Remote storage can be GCP bucket or local.

⸻

3. Model Development Pipeline

Includes:
	•	Baseline Logistic Regression
	•	Hyperparameter Tuning (GridSearch + SMOTE)
	•	LightGBM Model
	•	Best Model Selection
	•	Explainability (SHAP, LIME)
	•	Bias Detection across groups
	•	Drift Monitoring during new data arrival

Results stored in /models/ and /reports/.

⸻

4. CI/CD Pipeline (GitHub Actions)

Automatically runs on each push:

✔ Install dependencies
✔ Run DVC pull
✔ Train & Tune model
✔ Bias & Drift Checks
✔ Register best model
✔ Push final model to GCP Artifact Registry
✔ Upload reports as artifacts

File: .github/workflows/mlops_pipeline.yml

⸻

5. Model Registry (GCP Artifact Registry)

All final models are versioned using:

charlie-model-registry / charlie-mbta-model:version

Uploaded via gcloud and from CI/CD.

⸻

6. Model Deployment (Google Cloud Run)

The FastAPI service is deployed to:

https://charlie-mbta-api-XXXXXXXXXX.run.app

Runs the best model version.

⸻

7. Streamlit Frontend (User Interface)

Simple UI that calls the Cloud Run API and displays:
	•	Probability of Delay
	•	Predicted Outcome
	•	Model Version
	•	Confidence Score


⸻

Technologies Used

MLOps
	•	Apache Airflow
	•	DVC
	•	MLflow
	•	GitHub Actions
	•	GCP Artifact Registry
	•	Google Cloud Run

Modeling
	•	Scikit-learn
	•	LightGBM
	•	SMOTE
	•	SHAP, LIME
	•	Fairlearn (Bias detection)
	•	EvidentlyAI (Drift monitoring)

Deployment
	•	FastAPI
	•	Docker
	•	Cloud Run
	•	Streamlit

⸻

 Running the Project Locally

1. Clone the repository

git clone https://github.com/your-repo/charlie-mbta.git
cd charlie-mbta

2. Create virtual environment

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

3. Run Model Training

python -m ml_src.model_train

4. Run API locally

uvicorn ml_src.api.app:app --reload

5. Run Streamlit UI

streamlit run streamlit_app.py


⸻

 Cloud Run Deployment

gcloud builds submit --tag gcr.io/<project>/charlie-mbta-api
gcloud run deploy charlie-mbta-api \
    --image gcr.io/<project>/charlie-mbta-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated


⸻
 API Test (Local or Cloud Run)

curl -X POST "https://<cloud-run-url>/predict" \
  -H "Content-Type: application/json" \
  -d '{"direction_id": 0, "stop_sequence": 10}'


⸻

