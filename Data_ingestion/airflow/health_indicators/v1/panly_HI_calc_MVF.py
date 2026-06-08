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

DATA_SAVE_PATH = "/home/jpark/airflow/dags/calc_health_indicator/dataset/panly/"
NOMINAL_CELL_VOLTAGE = 4.2
TRAY = 20 
MAX_SOC = 65
MIN_SOC = 40
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
    


def calc_health_indicator_mvf(**context):
    dag_conf = context["dag_run"].conf
    filename = dag_conf.get("filename")
    # query_time = context["task_instance"].xcom_pull(task_ids="clac_past_days", key="query_time")
    # filename = str(query_time["start_time"])[:10]
    
    df = pd.read_feather(DATA_SAVE_PATH + filename + ".feather")

    nominal_rack_voltage = NOMINAL_CELL_VOLTAGE * TRAY * 12

    result_value = {}
    bank_list = df["BANK_ID"].unique()
    for bank_id in bank_list:
        result_value[int(bank_id)] = {}  # bank_id에 대한 내부 딕셔너리 생성
        filtered_df = df.query(f"BANK_ID == {bank_id}")
        rack_list = filtered_df["RACK_ID"].unique()

        for rack_id in rack_list:          
            # 데이터가 부족할 때는 어떻게 할것인가...?
            mvf_df = filtered_df.query(f"RACK_ID == {rack_id} and RACK_SOC >= {MIN_SOC}  and RACK_SOC <= {MAX_SOC} and BATTERY_STATUS_FOR_DISCHARGE == 1").reset_index(drop=True)
            mvf_df["RACK_VOLTAGE"] = abs(mvf_df["RACK_VOLTAGE"]-nominal_rack_voltage)

             # 컬럼이 아에 없을 때, iloc 접근이 안되므로... 
            try: 
                if ((MAX_SOC - 1) > mvf_df["RACK_SOC"].iloc[0]) or ((MIN_SOC + 1) < mvf_df["RACK_SOC"].iloc[-1]):
                    average = float(0)
                
                else:
                    # MVF 계산
                    average = round(float(mvf_df["RACK_VOLTAGE"].mean()), 2)
                
                min_soc = float(mvf_df["RACK_SOC"].min())
                max_soc = float(mvf_df["RACK_SOC"].max())
                min_vol = round(float(mvf_df["RACK_VOLTAGE"].min()), 2)
                max_vol = round(float(mvf_df["RACK_VOLTAGE"].max()), 2)

            except IndexError as e:
                average = float(0)
                min_soc = float(0)
                max_soc = float(0)
                min_vol = float(0)
                max_vol = float(0)
            
            result_value[bank_id][int(rack_id)] = {"mvf": average, 
                                                   "min_soc": min_soc, 
                                                   "max_soc": max_soc, 
                                                   "min_vol": min_vol, 
                                                   "max_vol": max_vol}
    
    context["task_instance"].xcom_push(key="result", value=result_value)
    print(result_value)



def push_data_to_database(**context):
    result = context["task_instance"].xcom_pull(task_ids="calc_mvf", key="result")
    query_time = context["task_instance"].xcom_pull(task_ids="clac_past_days", key="query_time")

    pg_hook = PostgresHook(postgres_conn_id='ess_stats')
    
    conn = pg_hook.get_conn()
    cur = conn.cursor()

    for bank, rack in result.items():
        bank_id = bank

        for rack_id, value in rack.items():
            print(f"bank_id = {bank_id}, rack_id = {rack_id}, value = {value}")
            insert_query = """INSERT INTO health_indicator_mvf ("TIMESTAMP", 
                                                                "OPERATING_SITE", 
                                                                "BANK_ID", 
                                                                "RACK_ID", 
                                                                "MVF", 
                                                                "MIN_SOC", 
                                                                "MAX_SOC", 
                                                                "MIN_VOLTAGE", 
                                                                "MAX_VOLTAGE") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            cur.execute(insert_query, (query_time["start_time"], OPERATING_SITE, bank_id, rack_id, value["mvf"], value["min_soc"], value["max_soc"], value["min_vol"], value["max_vol"]))
            conn.commit()
    
    cur.close()
    pg_hook.get_conn().close()



kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}

with DAG(dag_id='panly_HI_calc_MVF',
         default_args=default_args,
         start_date=datetime(2023, 1, 1, tzinfo=kst),
         schedule_interval=None,
         tags=['panly', 'indicator', 'rack'],
         catchup=True
         ) as dag:

    start = DummyOperator(task_id="start")
    
    clac_past_time = PythonOperator(
        task_id='clac_past_days',
        python_callable=clac_past_days
    )

    calc_mvf = PythonOperator(
        task_id='calc_mvf',
        python_callable=calc_health_indicator_mvf
    )

    push_data = PythonOperator(
        task_id='push_data',
        python_callable=push_data_to_database
    )
    end = DummyOperator(task_id="end")

start >> clac_past_time >> calc_mvf >> push_data >> end