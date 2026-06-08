from datetime import datetime, timedelta
from pytz import  timezone
from pprint import pprint
import pandas as pd
import os
import json
import copy
import pendulum

from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy import DummyOperator

DATA_SAVE_PATH = "/home/jpark/airflow/dags/calc_health_indicator/dataset/gold/"
OPERATING_SITE = 3

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
    


def calc_health_indicator_tiexvd(**context):
    dag_conf = context["dag_run"].conf
    filename = dag_conf.get("filename")
    
    df = pd.read_parquet(DATA_SAVE_PATH + filename + ".parquet")

    result_value = {}
    bank_list = df["BANK_ID"].unique()
    for bank_id in bank_list:
        result_value[int(bank_id)] = {}  # bank_id에 대한 내부 딕셔너리 생성
        filtered_df = df.query(f"BANK_ID == {bank_id}")
        rack_list = filtered_df["RACK_ID"].unique()

        for rack_id in rack_list:    
            #TIECVD
            try:           
                cvd_df = filtered_df.query(f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_CHARGE == 1").reset_index(drop=True)
                cvd_min_idx = cvd_df["RACK_VOLTAGE"].idxmin()
                cvd_max_idx = cvd_df["RACK_VOLTAGE"].idxmax()
                
                cvd_time_diff = abs(cvd_df.loc[cvd_min_idx, "TIMESTAMP"] - cvd_df.loc[cvd_max_idx, "TIMESTAMP"]).total_seconds()

                #TIEDVD
                dvd_df = filtered_df.query(f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_DISCHARGE == 1").reset_index(drop=True)
                dvd_min_idx = dvd_df["RACK_VOLTAGE"].idxmin()
                dvd_max_idx = dvd_df["RACK_VOLTAGE"].idxmax()
                dvd_time_diff = abs(dvd_df.loc[dvd_min_idx, "TIMESTAMP"] - dvd_df.loc[dvd_max_idx, "TIMESTAMP"]).total_seconds()
                
                result_value[bank_id][int(rack_id)] = {"cvd_time_diff": int(cvd_time_diff),
                                                    "cvd_min_vol": float(cvd_df["RACK_VOLTAGE"].min()), 
                                                    "cvd_max_vol": float(cvd_df["RACK_VOLTAGE"].max()), 
                                                    "dvd_time_diff": int(dvd_time_diff),     
                                                    "dvd_min_vol": float(dvd_df["RACK_VOLTAGE"].min()), 
                                                    "dvd_max_vol": float(dvd_df["RACK_VOLTAGE"].max())}
            except ValueError:
                print(f"Error: {bank_id}_{rack_id} no data for TIExvVD calculation")
                continue  # 다음 rack_id로 계속 진행

    context["task_instance"].xcom_push(key="result", value=result_value)
    print(result_value)


def push_data_to_database(**context):
    result = context["task_instance"].xcom_pull(task_ids="calc_tiexvd", key="result")
    query_time = context["task_instance"].xcom_pull(task_ids="clac_past_days", key="query_time")

    pg_hook = PostgresHook(postgres_conn_id='ess_stats')
    
    conn = pg_hook.get_conn()
    cur = conn.cursor()

    for bank, rack in result.items():
        bank_id = bank

        for rack_id, value in rack.items():
            print(f"bank_id = {bank_id}, rack_id = {rack_id}, value = {value}")
            insert_query = """INSERT INTO health_indicator_tiexvd ("TIMESTAMP", 
                                                                    "OPERATING_SITE", 
                                                                    "BANK_ID", 
                                                                    "RACK_ID", 
                                                                    "TIECVD", 
                                                                    "TIECVD_MIN_VOLTAGE", 
                                                                    "TIECVD_MAX_VOLTAGE", 
                                                                    "TIEDVD", 
                                                                    "TIEDVD_MIN_VOLTAGE", 
                                                                    "TIEDVD_MAX_VOLTAGE") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            cur.execute(insert_query, (query_time["start_time"], OPERATING_SITE, bank_id, rack_id, value["cvd_time_diff"], value["cvd_min_vol"], value["cvd_max_vol"], value["dvd_time_diff"], value["dvd_min_vol"], value["dvd_max_vol"]))
            conn.commit()
    
    cur.close()
    pg_hook.get_conn().close()


kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}

with DAG(dag_id='gold_HI_calc_TIExVD',
         default_args=default_args,
         start_date=datetime(2023, 11, 22, tzinfo=kst),
         schedule_interval=None,
         tags=['gold', 'indicator', 'rack'],
         catchup=True
         ) as dag:

    start = DummyOperator(task_id="start")
    
    clac_past_time = PythonOperator(
        task_id='clac_past_days',
        python_callable=clac_past_days
    )


    calc_tiexvd = PythonOperator(
        task_id='calc_tiexvd',
        python_callable=calc_health_indicator_tiexvd
    )

    push_data = PythonOperator(
        task_id='push_data',
        python_callable=push_data_to_database
    )

    end = DummyOperator(task_id="end")


start >> clac_past_time >> calc_tiexvd >> push_data >> end





