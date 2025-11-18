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
    dag_id="data_collection_dag",
    description="Stage 1: Collect MBTA route and schedule data",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@hourly",
    catchup=False,
    tags=["mbta", "stage1"],
) as dag:

    # Step 1: Fetch all routes
    fetch_routes = BashOperator(
        task_id="fetch_all_routes",
        bash_command=f"cd {PROJECT_DIR} && python scripts/fetch_all_routes.py",
    )

    # Step 2: Fetch schedule and prediction data
    fetch_data = BashOperator(
        task_id="fetch_data",
        bash_command=f"cd {PROJECT_DIR} && python scripts/fetch_data.py",
    )

    # Step 3: Trigger next stage (data processing DAG)
    trigger_processing = TriggerDagRunOperator(
        task_id="trigger_processing_dag",
        trigger_dag_id="data_processing_dag",
        reset_dag_run=True,
        wait_for_completion=False,
    )

    fetch_routes >> fetch_data >> trigger_processing