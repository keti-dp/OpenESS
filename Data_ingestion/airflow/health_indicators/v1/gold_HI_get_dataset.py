import pendulum

from datetime import datetime, timedelta
from pytz import  timezone

from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python_operator import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.dummy import DummyOperator


DATA_SAVE_PATH = "/home/jpark/airflow/dags/calc_health_indicator/dataset/gold/"

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
                rk."RACK_CURRENT", 
                rk."RACK_VOLTAGE",
                rk."RACK_MAX_CELL_VOLTAGE", 
                rk."RACK_MIN_CELL_VOLTAGE", 
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
                    "RACK_VOLTAGE",
                    "RACK_MAX_CELL_VOLTAGE",
                    "RACK_MIN_CELL_VOLTAGE"
                from rack
                where 
                    ("TIMESTAMP" between '{query_time["start_time"]}' and '{query_time["end_time"]}')
            ) AS rk 
            inner join (
                select
                    "TIMESTAMP",
                    "BATTERY_STATUS_FOR_STANDBY",
                    "BATTERY_STATUS_FOR_CHARGE",
                    "BATTERY_STATUS_FOR_DISCHARGE"
                from bank 
                where 
                    ("TIMESTAMP" between '{query_time["start_time"]}' and '{query_time["end_time"]}')
            ) as bk on rk."TIMESTAMP" = bk."TIMESTAMP" order by "TIMESTAMP" desc;
    """

    pg_hook = PostgresHook(postgres_conn_id='gold_site')
    df = pg_hook.get_pandas_df(sql=queries)
    df = df.sort_values(by=['TIMESTAMP', 'BANK_ID', 'RACK_ID'] ,ascending=True).reset_index(drop=True)
    
    df["TIMESTAMP"] = df["TIMESTAMP"].dt.tz_convert('Asia/Seoul')
    filename = str(query_time["start_time"])[:10]

    # TODO: Trigger를 사용하여 context["task_instance"].xcom_pull(task_ids=, key=, dag_id=) 형태로 받으려 했으나 받지 못하는 오류가 있음... 
    # 따라서, trigger의 conf를 사용함
        
    context["task_instance"].xcom_push(key="filename", value=filename)
    df.to_parquet(DATA_SAVE_PATH + filename + ".parquet")

    print(df)



kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}

with DAG(dag_id='gold_HI_get_dataset',
         default_args=default_args,
         start_date=datetime(2024, 4, 17, tzinfo=kst),
         schedule_interval='05 05 * * *',
         tags=['gold', 'indicator', 'rack', 'dataset'],
         catchup=True
         ) as dag:

    start = DummyOperator(task_id="start")
    
    clac_past_time = PythonOperator(
        task_id='clac_past_days',
        python_callable=clac_past_days
    )

    query_hi_dataset = PythonOperator(
        task_id='query_hi_dataset',
        python_callable=execute_select_query
    )

    trigger_TIExVD = TriggerDagRunOperator(
        task_id="trigger_TIExVD",
        trigger_dag_id="gold_HI_calc_TIExVD",
        execution_date="{{ execution_date }}",
        conf='{"filename": "{{ task_instance.xcom_pull(task_ids="query_hi_dataset", key="filename") }}"}'
    )

    trigger_PE = TriggerDagRunOperator(
        task_id="trigger_PE",
        trigger_dag_id="gold_HI_calc_PE",
        execution_date="{{ execution_date }}",
        conf='{"filename": "{{ task_instance.xcom_pull(task_ids="query_hi_dataset", key="filename") }}"}'
    )
    
    trigger_MVF = TriggerDagRunOperator(
        task_id="trigger_MVF",
        trigger_dag_id="gold_HI_calc_MVF",
        execution_date="{{ execution_date }}",
        conf='{"filename": "{{ task_instance.xcom_pull(task_ids="query_hi_dataset", key="filename") }}"}'
    )
    
    trigger_DCIR = TriggerDagRunOperator(
        task_id="trigger_DCIR",
        trigger_dag_id="gold_HI_calc_DCIR",
        execution_date="{{ execution_date }}",
        conf='{"filename": "{{ task_instance.xcom_pull(task_ids="query_hi_dataset", key="filename") }}"}'
    )

    trigger_VIExTD = TriggerDagRunOperator(
        task_id="trigger_VIExTD",
        trigger_dag_id="gold_HI_calc_VIExTD",
        execution_date="{{ execution_date }}",
        conf='{"filename": "{{ task_instance.xcom_pull(task_ids="query_hi_dataset", key="filename") }}"}'
    )
    end = DummyOperator(task_id="end")


start >> clac_past_time >> query_hi_dataset >>  [trigger_TIExVD, trigger_PE, trigger_MVF, trigger_DCIR, trigger_VIExTD] >> end
