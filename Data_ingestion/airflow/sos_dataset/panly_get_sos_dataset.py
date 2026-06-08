from datetime import datetime, timedelta
from pytz import  timezone
from pprint import pprint
import pendulum
import pandas as pd
import math
import os
from sqlalchemy import create_engine
from sqlalchemy import JSON

from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python_operator import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.dummy import DummyOperator

PATH = "/home/jpark/airflow/dags/get_sos_dataset/dataset/panly/"

SAFETY_INFO = {"RACK_MAX_CELL_VOLTAGE":{"upper_safety":4.086, "maximum_safety":4.076},
            "RACK_MIN_CELL_VOLTAGE":{"upper_safety":3.15, "maximum_safety":3.325},
            "RACK_CELL_VOLTAGE_GAP":{"upper_safety":0.3, "maximum_safety":0.158},
            "RACK_CURRENT":{"upper_safety":140, "maximum_safety":133},
            "RACK_MAX_CELL_TEMPERATURE":{"upper_safety":50, "maximum_safety":47.3},
            "RACK_MIN_CELL_TEMPERATURE":{"upper_safety":0, "maximum_safety":5.3},
            "RACK_CELL_TEMPERATURE_GAP":{"upper_safety":20, "maximum_safety":17.35}}

def tranfer_timezone(**context):
    # UTC 시간인 execution_date 
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
    query_time = context["task_instance"].xcom_pull(task_ids="transfer_tz", key="query_time")
    #query_time = {"start_time":"2022-05-01 00:00:00", "end_time":"2022-05-01 23:59:59"}
    queries = f"""
            SELECT 
                "TIMESTAMP", 
                "BANK_ID", 
                "RACK_ID", 
                "RACK_MAX_CELL_VOLTAGE", 
                "RACK_MIN_CELL_VOLTAGE", 
                "RACK_CELL_VOLTAGE_GAP",
                "RACK_CURRENT",
                "RACK_MAX_CELL_TEMPERATURE",
                "RACK_MIN_CELL_TEMPERATURE",
                "RACK_CELL_TEMPERATURE_GAP"
            FROM
                rack
            WHERE
                "TIMESTAMP" between '{query_time["start_time"]}' and '{query_time["end_time"]}'
            ORDER BY 
                "TIMESTAMP" ASC, "BANK_ID" ASC, "RACK_ID" ASC
    """

    pg_hook = PostgresHook(postgres_conn_id='panly_site')
    df = pg_hook.get_pandas_df(sql=queries)
    df = df.drop_duplicates(subset=['TIMESTAMP', 'BANK_ID', 'RACK_ID'], keep='last').reset_index(drop=True)
    
    if not df.empty:
        # 요거 안해주면 시간이 unixtime으로 나옴, timescaleDB에 따로 변환해주는게 없는듯....
        df["TIMESTAMP"] = df["TIMESTAMP"].dt.tz_convert('Asia/Seoul')

        filename = str(query_time["start_time"])[:10]
        df.to_feather(PATH + filename + ".feather")
        

def check_file_exists(**context):
    query_time = context["task_instance"].xcom_pull(task_ids="transfer_tz", key="query_time")
    filename = str(query_time["start_time"])[:10]
    file_path = PATH + filename + ".feather"
    
    if os.path.isfile(file_path):
        return 'get_sos_score'
    else:
        return 'end_task'
    


def sos_function(value, threshold, max_value):

    exp1 = 0.25 / math.pow(threshold - max_value, 2)   
    exp2 = math.pow(value - max_value, 2)
    exp3 = 1/(exp1*exp2+1)

    return exp3


def calc_sos_score(**context):
    query_time = context["task_instance"].xcom_pull(task_ids="transfer_tz", key="query_time")

    filename = str(query_time["start_time"])[:10]
    df = pd.read_feather(PATH + filename +".feather")
    df_dict = df.to_dict()

    f_safety = {"RACK_MAX_CELL_VOLTAGE":{},
                "RACK_MIN_CELL_VOLTAGE":{},
                "RACK_CELL_VOLTAGE_GAP":{},
                "RACK_CURRENT":{},
                "RACK_MAX_CELL_TEMPERATURE":{},
                "RACK_MIN_CELL_TEMPERATURE":{},
                "RACK_CELL_TEMPERATURE_GAP":{},
                "BANK_ID":{},
                "RACK_ID":{},
               }
    
    #각 tag_name(각 위험 요소)에 따른 safety 점수
    for tag_name in f_safety.keys():
        try:
            x_80 = SAFETY_INFO[tag_name]["upper_safety"]
            x_100 = SAFETY_INFO[tag_name]["maximum_safety"]
            
            for k, value in df_dict[tag_name].items():
                if x_100 - x_80 < 0:
                    if value < x_100:
                        f_safety[tag_name][k] = 1
                    else:
                        f_safety[tag_name][k] = sos_function(value, x_80, x_100)

                else:
                    if value > x_100:
                        f_safety[tag_name][k] = 1
                    else:
                        f_safety[tag_name][k] = sos_function(value, x_80, x_100)
        
        # BANK_ID, RACK_ID의 값을 넣기 위함 
        except KeyError:
            for k, value in df_dict[tag_name].items():
                f_safety[tag_name][k] = value           
    
    sos_df = pd.DataFrame(f_safety)
    # BANK_ID와 RACK_ID를 제외한 모든 컬럼 선택한 뒤, SoS 계산
    columns_to_multiply = sos_df.columns.difference(['BANK_ID', 'RACK_ID'])
    print(columns_to_multiply)
    sos_df['SOS_SCORE'] = sos_df[columns_to_multiply].prod(axis=1)

    # 최종 학습 데이터 셋
    final_df = pd.concat([df, sos_df["SOS_SCORE"]], axis=1)
    final_df.to_feather(PATH + filename + ".feather")
    print(final_df)



kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}

with DAG(dag_id='panly_get_sos_dataset',
         default_args=default_args,
         start_date=datetime(2024, 4, 17, tzinfo=kst),
         schedule_interval='00 4 * * *',
         tags=['dataset', 'sos'],
         catchup=True
         ) as dag:

    start = DummyOperator(task_id="start")
    
    transfer_tz = PythonOperator(
        task_id='transfer_tz',
        python_callable=tranfer_timezone
    )
    
    get_data = PythonOperator(
        task_id='get_data',
        python_callable=execute_select_query
    )

    check_file = BranchPythonOperator(
       task_id='check_file',
        python_callable=check_file_exists,
        provide_context=True
    )


    get_sos_score = PythonOperator(
        task_id='get_sos_score',
        python_callable=calc_sos_score
    )

    end_task = DummyOperator(
        task_id='end_task'
    )

start >> transfer_tz >> get_data >> check_file
check_file >>  get_sos_score >> end_task
check_file >> end_task