# 🚀 Charlie MBTA Model Deployment Guide

## Overview

Charlie is an AI-powered MBTA transit assistant that provides real-time predictions, alerts, and natural language interaction for Boston's public transportation system.

### Key Features
- 🤖 **AI Chatbot** - Natural language queries powered by OpenAI GPT-4
- 🚇 **Real-Time Predictions** - Live data from MBTA API
- 📊 **ML Model** - Trained model for transit predictions
- 🔄 **Auto-Retraining** - Automatic retraining on data drift
- 📈 **Monitoring** - MLflow tracking and drift detection
- ☁️ **Cloud Deployment** - GCP Cloud Run with Docker

### Tech Stack
| Component | Technology |
|-----------|------------|
| Backend | Flask (Python 3.12) |
| AI/ML | OpenAI GPT-4, scikit-learn, LightGBM |
| Cloud | Google Cloud Platform (Cloud Run) |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| ML Tracking | MLflow |
| Data Pipeline | Apache Airflow, DVC |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Repository                         │
│                    github.com/Gnanasudharsan/Charlie            │
└─────────────────────┬───────────────────────────────────────────┘
                      │ Push to main
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GitHub Actions CI/CD                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │    Test     │→ │    Build    │→ │  Deploy to Cloud Run    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GCP Cloud Run                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Charlie MBTA API Service                    │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────────────────┐  │    │
│  │  │  Flask    │ │  OpenAI   │ │    ML Model           │  │    │
│  │  │  Server   │ │  Client   │ │  (final_model.joblib) │  │    │
│  │  └───────────┘ └───────────┘ └───────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External APIs                                 │
│  ┌─────────────────┐          ┌─────────────────────────────┐   │
│  │    MBTA API     │          │      OpenAI API             │   │
│  │  (Real-time)    │          │   (Natural Language)        │   │
│  └─────────────────┘          └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Required Tools
```bash
# Check versions
python --version    # Python 3.11+
docker --version    # Docker 20.10+
gcloud --version    # Google Cloud SDK
git --version       # Git 2.0+
```

### Required Accounts
- [Google Cloud Platform](https://console.cloud.google.com/) account with billing enabled
- [OpenAI](https://platform.openai.com/) API key
- [MBTA](https://api-v3.mbta.com/register) API key 
- [GitHub](https://github.com/) account

### GCP Setup
```bash
# Install Google Cloud SDK (if not installed)
# macOS
brew install google-cloud-sdk

# Ubuntu/Debian
sudo apt-get install google-cloud-sdk

# Authenticate
gcloud auth login
gcloud config set project charlie-478223
```

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Gnanasudharsan/Charlie.git
cd Charlie
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
# Create .env file
cat > .env << EOF
MBTA_API_KEY=your_mbta_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
EOF
```

### 4. Run Locally
```bash
cd API_Service
python app.py
# Open http://localhost:5001
```

---

## Cloud Deployment (GCP)

### Option 1: Automated Deployment (Recommended)

Push to `main` branch triggers automatic deployment via GitHub Actions.

```bash
git add .
git commit -m "Deploy update"
git push origin main
```

### Option 2: Manual Deployment

#### Step 1: Authenticate with GCP
```bash
gcloud auth login
gcloud config set project charlie-478223
gcloud auth configure-docker
```

#### Step 2: Build Docker Image
```bash
cd API_Service

docker build \
  --platform linux/amd64 \
  -t gcr.io/charlie-478223/charlie-mbta-chatbot:latest .
```

#### Step 3: Push to Container Registry
```bash
docker push gcr.io/charlie-478223/charlie-mbta-chatbot:latest
```

#### Step 4: Deploy to Cloud Run
```bash
gcloud run deploy charlie-mbta-chatbot \
  --image gcr.io/charlie-478223/charlie-mbta-chatbot:latest \
  --platform managed \
  --region us-east1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 10 \
  --update-env-vars "MBTA_API_KEY=your_key" \
  --update-env-vars "OPENAI_API_KEY=your_key"
```

#### Step 5: Verify Deployment
```bash
# Get service URL
gcloud run services describe charlie-mbta-chatbot \
  --region us-east1 \
  --format 'value(status.url)'

# Test health endpoint
curl https://charlie-mbta-chatbot-588293495748.us-east1.run.app/health
```

---

## CI/CD Pipeline

### Pipeline Overview

```yaml
Trigger: Push to main branch
    │
    ├── Job 1: Test
    │   ├── Checkout code
    │   ├── Setup Python 3.11
    │   ├── Install dependencies
    │   └── Run health checks
    │
    ├── Job 2: Build & Deploy
    │   ├── Authenticate to GCP
    │   ├── Build Docker image
    │   ├── Push to GCR
    │   └── Deploy to Cloud Run
    │
    └── Job 3: Notify
        └── Send deployment status
```

### GitHub Secrets Required

Configure these in `Settings > Secrets > Actions`:

| Secret Name | Description |
|-------------|-------------|
| `GCP_SA_KEY` | GCP Service Account JSON key |
| `OPENAI_API_KEY` | OpenAI API key |
| `MBTA_API_KEY` | MBTA API key |
| `SLACK_WEBHOOK_URL` | (Optional) Slack notifications |



---

## Model Monitoring & Retraining

### Monitoring Components

| Component | File | Purpose |
|-----------|------|---------|
| Drift Detection | `Model_Development/ml_src/monitor_drift.py` | Detects data distribution shifts |
| Bias Analysis | `Model_Development/ml_src/bias_analysis.py` | Monitors model fairness |
| Auto-Retrain | `retraining/trigger_retrain.py` | Triggers retraining pipeline |
| Notifications | `retraining/notifier.py` | Sends alerts via Slack/Email |

### Drift Detection

```bash
# Run drift monitoring
python -m Model_Development.ml_src.monitor_drift

# Check drift report
cat reports/drift_report.json
```

### Automatic Retraining

The system automatically triggers retraining when:
- Model accuracy drops below threshold (default: 0.85)
- Data drift PSI > 0.2
- Manual trigger via scheduled job

```bash
# Manual trigger
python -m retraining.trigger_retrain

# With notifications
python -m retraining.trigger_retrain_with_notify

# Scheduled (cron)
./retraining/run_scheduled.sh
```

### Retraining Pipeline Flow

```
Drift Detected → Trigger Retrain → Train New Model → Validate
                                                        │
                                    ┌───────────────────┴───────────────────┐
                                    │                                       │
                              Performance Better?                    Performance Worse?
                                    │                                       │
                                    ▼                                       ▼
                            Deploy New Model                      Keep Current Model
                                    │                                       │
                                    └───────────────────┬───────────────────┘
                                                        │
                                                        ▼
                                                Send Notification
```

---

## API Endpoints

### Base URL
```
Production: https://charlie-mbta-chatbot-588293495748.us-east1.run.app
Local: http://localhost:5001
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main dashboard UI |
| GET | `/health` | Health check |
| POST | `/api/chat` | AI chatbot |
| GET | `/chat?msg=` | AI chatbot (GET) |
| GET | `/api/routes` | All MBTA routes |
| GET | `/api/alerts` | Service alerts |
| GET | `/api/predictions?stop=` | Real-time predictions |
| GET | `/api/vehicles?route=` | Vehicle positions |
| GET | `/api/stops?route=` | Stops for a route |
| GET | `/api/dashboard` | Dashboard overview |
| GET | `/api/line/<line_id>` | Line status |
| GET | `/api/station/<stop_id>` | Station info |


---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MBTA_API_KEY` | Yes | MBTA API key for real-time data |
| `OPENAI_API_KEY` | Yes | OpenAI API key for AI chatbot |
| `PORT` | No | Server port (default: 8080) |
| `FLASK_ENV` | No | Environment (production/development) |
| `SLACK_WEBHOOK_URL` | No | Slack webhook for notifications |

---

## Project Structure

```
Charlie/
├── API_Service/                 # 🚀 Deployment Service
│   ├── app.py                   # Flask application
│   ├── Dockerfile               # Container configuration
│   ├── requirements.txt         # Python dependencies
│   ├── templates/               # HTML templates
│   ├── static/                  # CSS/JS files
│   └── models/                  # ML model files
│       └── final_model.joblib
│
├── Model_Development/           # 🧠 ML Pipeline
│   ├── ml_src/
│   │   ├── model_train.py       # Model training
│   │   ├── model_tuning.py      # Hyperparameter tuning
│   │   ├── model_select.py      # Model selection
│   │   ├── model_explain.py     # SHAP/LIME explainability
│   │   ├── bias_analysis.py     # Fairness analysis
│   │   ├── monitor_drift.py     # Drift detection
│   │   ├── register_model.py    # MLflow registration
│   │   └── gcp_registry.py      # GCP Artifact Registry
│   └── reports/                 # Generated reports
│
├── Data_Pipeline/               # 📊 Data Pipeline
│   ├── dags/                    # Airflow DAGs
│   └── data/                    # Raw & processed data
│
├── retraining/                  # 🔄 Auto-Retraining
│   ├── trigger_retrain.py       # Retrain trigger
│   ├── trigger_retrain_with_notify.py
│   ├── notifier.py              # Slack/Email alerts
│   └── run_scheduled.sh         # Cron script
│
├── .github/workflows/           # ⚙️ CI/CD
│   └── deploy.yml               # GitHub Actions
│
├── mlruns/                      # 📈 MLflow tracking
├── models/                      # 💾 Trained models
├── reports/                     # 📋 Analysis reports
└── requirements.txt             # Project dependencies
```

---


## Video Demo

### Recording Checklist

Your deployment video (5-10 minutes) should demonstrate:

- [ ] **Fresh Environment Setup**
  - Start from clean terminal
  - Show no prior installations

- [ ] **Clone & Setup**
  ```bash
  git clone https://github.com/Gnanasudharsan/Charlie.git
  cd Charlie
  pip install -r API_Service/requirements.txt
  ```

- [ ] **Docker Build**
  ```bash
  cd API_Service
  docker build --platform linux/amd64 -t gcr.io/charlie-478223/charlie-mbta-chatbot:latest .
  ```

- [ ] **Deploy to GCP**
  ```bash
  docker push gcr.io/charlie-478223/charlie-mbta-chatbot:latest
  gcloud run deploy charlie-mbta-chatbot ...
  ```

- [ ] **Verify Deployment**
  - Access deployed URL
  - Test chatbot functionality
  - Show health endpoint

- [ ] **Show CI/CD**
  - Make a code change
  - Push to GitHub
  - Show automatic deployment

---

## Links

| Resource | URL |
|----------|-----|
| **Live Application** | https://charlie-mbta-chatbot-588293495748.us-east1.run.app |
| **GitHub Repository** | https://github.com/Gnanasudharsan/Charlie |
| **MBTA API Docs** | https://api-v3.mbta.com/docs/swagger |
| **OpenAI API** | https://platform.openai.com |

---

## Author

**Gnanasudharsan Ashokumar**

- GitHub: [@Gnanasudharsan](https://github.com/Gnanasudharsan)
- Project: Charlie MBTA Transit Assistant

---

