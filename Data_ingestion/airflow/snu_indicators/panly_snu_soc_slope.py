from datetime import datetime, timedelta
from pytz import  timezone
import pandas as pd
import pendulum

from airflow import DAG
from airflow.operators.python_operator import BranchPythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy import DummyOperator

DATA_SAVE_PATH = "/home/jpark/airflow/dags/calc_snu_indicator/dataset/panly/"
OPERATING_SITE = 2

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
    


def calc_snu_soc_slope(**context):
    dag_conf = context["dag_run"].conf
    filename = dag_conf.get("filename")
    
    df = pd.read_parquet(DATA_SAVE_PATH + filename + ".parquet")

    soc_result = []
    bank_list = df["BANK_ID"].unique()
    skip_push = True  # 기본적으로 push를 생략하도록 설정

    for bank_id in bank_list:
        bank_df = df[df["BANK_ID"] == bank_id]

        for rack_id in bank_df["RACK_ID"].unique():    
            
            try:           
                filtered_df = bank_df[bank_df["RACK_ID"] == rack_id].reset_index(drop=True)
                
                # HTC-충남대 데이터는 0~100프로 까지 데이터가 존재하지만, 운영사이트는 없기 때문에 
                # 아래와 같이 segments를 가변적으로 둠
                n_segments = int((filtered_df["RACK_SOC"].max() - filtered_df["RACK_SOC"].min()) // 10)

                try:
                    segment_length = len(filtered_df["RACK_SOC"]) // n_segments + 1
                
                except ZeroDivisionError:
                    print(f"The ESS is not operating, so there is no change in SOC")
                    continue

                soc_segments = [filtered_df["RACK_SOC"][i:i+segment_length] for i in range(0, len(filtered_df["RACK_SOC"]), segment_length)]
                time_segments = [filtered_df["TIMESTAMP"][i:i+segment_length] for i in range(0, len(filtered_df["TIMESTAMP"]), segment_length)]
                slopes = []
                for soc_seg, time_seg in zip(soc_segments, time_segments):
                    if len(soc_seg) > 1:
                        time_diff = (time_seg.iloc[-1] - time_seg.iloc[0]).total_seconds() 
                        slope = (soc_seg.iloc[-1] - soc_seg.iloc[0]) / time_diff
                        slopes.append(slope)

                max_slope = min(slopes) * 100 # 기존 segment 244 이지만, 운영사이트는 2285임(테스트 날짜 기준) 따라서, 곱하기 10 -> 100 
                soc_result.append((int(bank_id), int(rack_id), float(round(max_slope, 4))))
                skip_push = False  # 데이터가 있으므로 push 작업 수행

            except ValueError:
                print(f"Error: {bank_id}_{rack_id} no data for SoC slope calculation")
                continue  # 다음 rack_id로 계속 진행

    context["task_instance"].xcom_push(key="result", value=soc_result)
    context["task_instance"].xcom_push(key="skip_push", value=skip_push)
    print(soc_result)


def choose_push_or_end(**context):
    skip_push = context["task_instance"].xcom_pull(task_ids="calc_soc_slope", key="skip_push")
    return "end" if skip_push else "push_data"


def push_data_to_database(**context):
    result = context["task_instance"].xcom_pull(task_ids="calc_soc_slope", key="result")
    query_time = context["task_instance"].xcom_pull(task_ids="clac_past_days", key="query_time")
    
    pg_hook = PostgresHook(postgres_conn_id='ess_stats')
    
    conn = pg_hook.get_conn()
    cur = conn.cursor()

    for item in result:
        
        print(f"bank_id = {item[0]}, rack_id = {item[1]}, value = {item[2]}")
        insert_query = """INSERT INTO snu_soc_slope ("TIMESTAMP", 
                                                    "OPERATING_SITE", 
                                                    "BANK_ID", 
                                                    "RACK_ID", 
                                                    "SOC_SLOPE") VALUES (%s, %s, %s, %s, %s)"""
        cur.execute(insert_query, (query_time["start_time"], OPERATING_SITE, item[0], item[1], item[2]))
        conn.commit()
    
    cur.close()
    pg_hook.get_conn().close()


kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}

with DAG(dag_id='panly_snu_soc_slope',
         default_args=default_args,
         start_date=datetime(2024, 1, 1, tzinfo=kst),
         schedule_interval=None,
         tags=['panly', 'indicator', 'snu'],
         catchup=True
         ) as dag:

    start = DummyOperator(task_id="start")
    
    clac_past_time = PythonOperator(
        task_id='clac_past_days',
        python_callable=clac_past_days
    )


    calc_soc_slope = PythonOperator(
        task_id='calc_soc_slope',
        python_callable=calc_snu_soc_slope
    )

    branch = BranchPythonOperator(
        task_id='choose_push_or_end',
        python_callable=choose_push_or_end
    )

    push_data = PythonOperator(
        task_id='push_data',
        python_callable=push_data_to_database
    )

    end = DummyOperator(task_id="end")


start >> clac_past_time >> calc_soc_slope >> branch 
branch >> push_data 
branch >> end





