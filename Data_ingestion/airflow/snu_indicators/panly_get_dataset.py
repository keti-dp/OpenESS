from datetime import datetime, timedelta
from pytz import  timezone
from pprint import pprint
import pendulum
import os 
from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python_operator import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.dummy import DummyOperator

DATA_SAVE_PATH = "/home/jpark/airflow/dags/calc_snu_indicator/dataset/panly/"

def clac_past_days(**context):
    execution_date = context["execution_date"]

    seoul_tz = timezone('Asia/Seoul')
    execution_date_seoul = execution_date.astimezone(seoul_tz)

    print(f"Excution_data: {execution_date}")
    print(f"execution_date_seoul: {execution_date_seoul}")
      
    start_time = execution_date_seoul.start_of('day').to_datetime_string()
    end_time = execution_date_seoul.end_of('day').to_datetime_string()

    query_time = {"start_time":start_time, "end_time":end_time}
    print(query_time)
    context["task_instance"].xcom_push(key="query_time", value=query_time)
    

def execute_select_query(**context):
    query_time = context["task_instance"].xcom_pull(task_ids="clac_past_days", key="query_time")
    
    queries = f"""
                select DISTINCT
                    rk."TIMESTAMP",
                    rk."BANK_ID",
                    rk."RACK_ID",
                    rk."RACK_SOC",
                    rk."RACK_MAX_CELL_VOLTAGE",
                    bk."BATTERY_STATUS_FOR_CHARGE",
                    bk."BATTERY_STATUS_FOR_STANDBY",
                    bk."BATTERY_STATUS_FOR_DISCHARGE"
                from (
                    select
                        "TIMESTAMP",
                        "BANK_ID",
                        "RACK_ID",
                        "RACK_SOC",
                        "RACK_MAX_CELL_VOLTAGE"
                    from rack
                    where 
                        ("TIMESTAMP" between '{query_time["start_time"]}' and '{query_time["end_time"]}')
                ) AS rk 
                inner join (
                    select
                        "TIMESTAMP",
                        "BATTERY_STATUS_FOR_CHARGE",
                        "BATTERY_STATUS_FOR_STANDBY",
                        "BATTERY_STATUS_FOR_DISCHARGE"
                    from bank 
                    where 
                        ("TIMESTAMP" between '{query_time["start_time"]}' and '{query_time["end_time"]}')
                ) as bk on rk."TIMESTAMP" = bk."TIMESTAMP" order by "TIMESTAMP" asc;
        """

    pg_hook = PostgresHook(postgres_conn_id='panly_site')
    df = pg_hook.get_pandas_df(sql=queries)
    df = df.sort_values(by=['TIMESTAMP', 'BANK_ID', 'RACK_ID'] ,ascending=True).reset_index(drop=True)
    
    filename = str(query_time["start_time"])[:10]
    try:
        df["TIMESTAMP"] = df["TIMESTAMP"].dt.tz_convert('Asia/Seoul')
    
        # TODO: Trigger를 사용하여 context["task_instance"].xcom_pull(task_ids=, key=, dag_id=) 형태로 받으려 했으나 받지 못하는 오류가 있음... 
        # 따라서, trigger의 conf를 사용함
        
        context["task_instance"].xcom_push(key="filename", value=filename)
        df.to_parquet(DATA_SAVE_PATH + filename + ".parquet")

        print(df)
    except AttributeError:
        print(f"{filename} No Data")


def calc_snu_indciator(**context):
    result_value = context["task_instance"].xcom_pull(task_ids='get_snu_dataset', key='filename')

    if result_value:
        return ['branch_check_files', 'trigger_soc_slope']
    else:
        return 'end'


def check_files_count(**context):
    file_list = [f for f in os.listdir(DATA_SAVE_PATH) if f.endswith('.parquet')]
    
    if len(file_list) >= 90:
        return 'trigger_voltage_relax'
    else:
        return 'end'
    


kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}

with DAG(dag_id='panly_snu_indicator_get_dataset',
         default_args=default_args,
         start_date=datetime(2024, 1, 1, tzinfo=kst),
         schedule_interval='00 12 * * *',
         tags=['panly', 'indicator', 'dataset', 'snu'],
         catchup=True
         ) as dag:

    start = DummyOperator(task_id="start")
    
    clac_past_time = PythonOperator(
        task_id='clac_past_days',
        python_callable=clac_past_days
    )

    get_snu_dataset = PythonOperator(
        task_id='get_snu_dataset',
        python_callable=execute_select_query
    )

    branch_snu_indicator = BranchPythonOperator(
        task_id='branch_snu_indicator',
        python_callable=calc_snu_indciator,
        provide_context=True,
        dag=dag,
    )

    branch_check_files = BranchPythonOperator(
        task_id='branch_check_files',
        python_callable=check_files_count,
        provide_context=True,
        dag=dag
    )

    trigger_voltage_relax = TriggerDagRunOperator(
        task_id="trigger_voltage_relax",
        trigger_dag_id="panly_snu_voltage_relax",
        execution_date="{{ execution_date }}",
        conf='{"filename": "{{ task_instance.xcom_pull(task_ids="get_snu_dataset", key="filename") }}"}'
    )

    trigger_soc_slope = TriggerDagRunOperator(
        task_id="trigger_soc_slope",
        trigger_dag_id="panly_snu_soc_slope",
        execution_date="{{ execution_date }}",
        conf='{"filename": "{{ task_instance.xcom_pull(task_ids="get_snu_dataset", key="filename") }}"}'
    )
    
    end = DummyOperator(task_id="end")
    complete_snu_indicator = DummyOperator(task_id="complete_snu_indicators")


start >> clac_past_time >> get_snu_dataset >> branch_snu_indicator
branch_snu_indicator >> branch_check_files
branch_check_files >> trigger_voltage_relax >> complete_snu_indicator
branch_snu_indicator >> trigger_soc_slope >> complete_snu_indicator
branch_snu_indicator >> end
branch_check_files >> end