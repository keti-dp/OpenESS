from datetime import datetime, timedelta
from pytz import  timezone
from pprint import pprint
import pendulum
import pandas as pd
import os
from sqlalchemy import create_engine
from sqlalchemy import JSON

from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python_operator import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.dummy import DummyOperator

OUTPUT_PATH = "/home/jpark/airflow/dags/calc_socp/dataset/panly/"

def tranfer_timezone(**context):
    # UTC 시간인 execution_date 
    execution_date = context["execution_date"]

    seoul_tz = timezone('Asia/Seoul')
    execution_date_seoul = execution_date.astimezone(seoul_tz)

    print(f"Excution_data: {execution_date}")
    print(f"execution_date_seoul: {execution_date_seoul}")
    begin_time = execution_date_seoul.start_of('day').to_datetime_string()
    end_time = execution_date_seoul.end_of('day').to_datetime_string()
    
    query_time = {"begin_time":begin_time, "end_time":end_time}
    print(query_time)

    context["task_instance"].xcom_push(key="query_time", value=query_time)


    

def execute_select_query(**context):
    query_time = context["task_instance"].xcom_pull(task_ids="transfer_tz", key="query_time")
    
    queries = f"""
            select DISTINCT
                rk."TIMESTAMP",
                rk."BANK_ID",
                rk."RACK_ID",
                rk."RACK_SOC", 
                rk."RACK_CURRENT", 
                rk."RACK_MAX_CELL_VOLTAGE", 
                rk."RACK_MIN_CELL_VOLTAGE", 
                rk."RACK_MAX_CELL_VOLTAGE_POSITION", 
                rk."RACK_MIN_CELL_VOLTAGE_POSITION", 
                rk."RACK_MAX_CELL_TEMPERATURE", 
                rk."RACK_MIN_CELL_TEMPERATURE", 
                rk."RACK_MAX_CELL_TEMPERATURE_POSITION", 
                rk."RACK_MIN_CELL_TEMPERATURE_POSITION", 
                bk."BATTERY_STATUS_FOR_STANDBY", 
                bk."BATTERY_STATUS_FOR_CHARGE", 
                bk."BATTERY_STATUS_FOR_DISCHARGE"
            from (
                select
                    "TIMESTAMP",
                    "BANK_ID",
                    "RACK_ID",
                    "RACK_SOC",
                    "RACK_CURRENT",
                    "RACK_MAX_CELL_VOLTAGE",
                    "RACK_MIN_CELL_VOLTAGE",
                    "RACK_MAX_CELL_VOLTAGE_POSITION",
                    "RACK_MIN_CELL_VOLTAGE_POSITION",
                    "RACK_MAX_CELL_TEMPERATURE",
                    "RACK_MIN_CELL_TEMPERATURE",
                    "RACK_MAX_CELL_TEMPERATURE_POSITION",
                    "RACK_MIN_CELL_TEMPERATURE_POSITION"
                from rack
                where 
                    ("TIMESTAMP" between '{query_time["begin_time"]}' and '{query_time["end_time"]}')
            ) AS rk 
            inner join (
                select
                    "TIMESTAMP",
                    "BATTERY_STATUS_FOR_STANDBY",
                    "BATTERY_STATUS_FOR_CHARGE",
                    "BATTERY_STATUS_FOR_DISCHARGE"
                from bank 
                where 
                    ("TIMESTAMP" between '{query_time["begin_time"]}' and '{query_time["end_time"]}')
            ) as bk on rk."TIMESTAMP" = bk."TIMESTAMP" order by "TIMESTAMP" desc;
    """

    pg_hook = PostgresHook(postgres_conn_id='panly_site')
    df = pg_hook.get_pandas_df(sql=queries)
    df = df.sort_values(by=['TIMESTAMP', 'BANK_ID', 'RACK_ID'] ,ascending=True).reset_index(drop=True)
    
    filename = str(query_time["begin_time"])[:10]
    try:
        df["TIMESTAMP"] = df["TIMESTAMP"].dt.tz_convert('Asia/Seoul')
    
        context["task_instance"].xcom_push(key="filename", value=filename)
        df.to_feather(OUTPUT_PATH + filename + ".feather")   
    except AttributeError:
        print(f"{filename} No Data")

def check_data_for_push(**context):
    result_value = context["task_instance"].xcom_pull(task_ids='get_data', key='filename')
    print(result_value)
    if result_value:
        return ['trigger_count_socp', 'trigger_info_socp']
    else:
        return 'end'        


kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}

with DAG(dag_id='panly_GetDataset_SoCP',
         default_args=default_args,
         start_date=datetime(2024, 4, 17, tzinfo=kst),
         schedule_interval='00 4 * * *',
         tags=['panly', 'dataset', 'socp'],
         catchup=False
         ) as dag:

    start = DummyOperator(task_id="start")
    
    transfer_tz = PythonOperator(
        task_id='transfer_tz',
        python_callable=tranfer_timezone
    )
    
    get_data = PythonOperator(
        task_id='get_data',
        python_callable=execute_select_query,
    )

    branch_operator = BranchPythonOperator(
        task_id='branch_check_data',
        python_callable=check_data_for_push,
        provide_context=True,
        dag=dag,
    )

    trigger_count_socp = TriggerDagRunOperator(
        task_id = "trigger_count_socp",
        trigger_dag_id = "panly_count_socp",
        execution_date = "{{ execution_date }}",
        conf = '{"filename": "{{ task_instance.xcom_pull(task_ids="get_data", key="filename") }}"}'
    )

    trigger_info_socp = TriggerDagRunOperator(
        task_id = "trigger_info_socp",
        trigger_dag_id = "panly_info_socp",
        execution_date = "{{ execution_date }}",
        conf = '{"filename": "{{ task_instance.xcom_pull(task_ids="get_data", key="filename") }}"}'
    )

    end = DummyOperator(task_id="end")
    complete_socp = DummyOperator(task_id="complete_socp")

start >> transfer_tz >> get_data >> branch_operator
branch_operator >> [trigger_count_socp, trigger_info_socp] >> complete_socp
branch_operator >> end
