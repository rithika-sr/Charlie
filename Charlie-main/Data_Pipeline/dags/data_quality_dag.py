from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# ✅ Inside Docker
PROJECT_DIR = "/opt/airflow/Data_Pipeline"

default_args = {
    "owner": "charlie",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="data_quality_dag",
    description="Stage 3: Detect anomalies and perform quality checks on MBTA data",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mbta", "stage3"],
) as dag:

    detect_anomalies = BashOperator(
        task_id="detect_anomalies",
        bash_command=f"cd {PROJECT_DIR} && python scripts/detect_anomalies.py",
    )

    run_quality_checks = BashOperator(
        task_id="run_quality_checks",
        bash_command=f"cd {PROJECT_DIR} && python scripts/run_quality_checks.py",
    )

    detect_anomalies >> run_quality_checks