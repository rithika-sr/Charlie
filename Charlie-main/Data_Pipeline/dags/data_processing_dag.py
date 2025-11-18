from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# ✅ Inside Docker
PROJECT_DIR = "/opt/airflow/Data_Pipeline"

default_args = {
    "owner": "charlie",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="data_processing_dag",
    description="Stage 2: Preprocess and validate MBTA data",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mbta", "stage2"],
) as dag:

    # Step 1: Preprocess raw MBTA JSONs
    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command=f"cd {PROJECT_DIR} && python scripts/preprocess_data.py",
    )

    # Step 2: Validate schema using your schema validation script
    validate_schema = BashOperator(
        task_id="validate_schema",
        bash_command=f"cd {PROJECT_DIR} && python scripts/validate_schema.py",
    )

    # Step 3: Trigger data quality DAG
    trigger_quality = TriggerDagRunOperator(
        task_id="trigger_quality_dag",
        trigger_dag_id="data_quality_dag",
        reset_dag_run=True,
        wait_for_completion=False,
    )

    preprocess >> validate_schema >> trigger_quality