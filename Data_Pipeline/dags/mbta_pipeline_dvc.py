from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# ✅ Inside Docker — consistent with your working setup
PROJECT_DIR = "/opt/airflow/Data_Pipeline"

default_args = {
    'owner': 'charlie',
    'depends_on_past': False,
    'email': ['ashokumar.g@northeastern.edu'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id="mbta_final_data_pipeline",
    description="Unified MBTA data pipeline (collection → processing → quality)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",   # change to @hourly when testing
    catchup=False,
    tags=["mbta", "final_pipeline"],
) as dag:

    # ─────────────────────────────
    # STAGE 1 — Data Collection
    # ─────────────────────────────
    fetch_all_routes = BashOperator(
        task_id="fetch_all_routes",
        bash_command=f"cd {PROJECT_DIR} && python scripts/fetch_all_routes.py",
    )

    fetch_data = BashOperator(
        task_id="fetch_data",
        bash_command=f"cd {PROJECT_DIR} && python scripts/fetch_data.py",
    )

    # ─────────────────────────────
    # STAGE 2 — Data Processing
    # ─────────────────────────────
    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command=f"cd {PROJECT_DIR} && python scripts/preprocess_data.py",
    )

    validate_schema = BashOperator(
        task_id="validate_schema",
        bash_command=f"cd {PROJECT_DIR} && python scripts/validate_schema.py",
    )

    # ─────────────────────────────
    # STAGE 3 — Data Quality & Anomaly Detection
    # ─────────────────────────────
    detect_anomalies = BashOperator(
        task_id="detect_anomalies",
        bash_command=f"cd {PROJECT_DIR} && python scripts/detect_anomalies.py",
    )

    run_quality_checks = BashOperator(
        task_id="run_quality_checks",
        bash_command=f"cd {PROJECT_DIR} && python scripts/run_quality_checks.py",
    )

    # ─────────────────────────────
    # TASK DEPENDENCIES
    # ─────────────────────────────
    (
        fetch_all_routes
        >> fetch_data
        >> preprocess_data
        >> validate_schema
        >> detect_anomalies
        >> run_quality_checks
    )