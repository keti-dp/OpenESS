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
from airflow.operators.python_operator import BranchPythonOperator

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
    


def calc_health_indicator_viextd(**context):
    dag_conf = context["dag_run"].conf
    filename = dag_conf.get("filename")
    
    df = pd.read_parquet(DATA_SAVE_PATH + filename + ".parquet")

    result_value = {}
    soc = 60
    time = 1000 
    bank_list = df["BANK_ID"].unique()
    for bank_id in bank_list:
        result_value[int(bank_id)] = {}  # bank_id에 대한 내부 딕셔너리 생성
        filtered_df = df.query(f"BANK_ID == {bank_id}")
        rack_list = filtered_df["RACK_ID"].unique()

        for rack_id in rack_list:
            try:
                # rack_id에 대한 내부 딕셔너리 생성
                result_value[int(bank_id)][int(rack_id)] = {}  

                # 데이터가 부족할 때는 어떻게 할것인가...?
                viectd_df = filtered_df.query(f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_CHARGE == 1").reset_index(drop=True)
                viedtd_df = filtered_df.query(f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_DISCHARGE == 1").reset_index(drop=True)

                # 'TIMESTAMP'를 index로 설정 및 중복제거
                viectd_df = viectd_df.set_index('TIMESTAMP')
                viedtd_df = viedtd_df.set_index('TIMESTAMP')
                viectd_df = viectd_df.loc[~viectd_df.index.duplicated(keep='first')]
                viedtd_df = viedtd_df.loc[~viedtd_df.index.duplicated(keep='first')]
                
                # SOC 60%인 데이터, 없다면 근처 데이터를 쿼리
                if viectd_df.query(f"RACK_SOC == {soc}").empty:
                    closest_row = (viectd_df['RACK_SOC'] - soc).abs().idxmin()
                    viectd_soc = viectd_df.loc[[closest_row]]
                else:
                    viectd_soc = viectd_df.query(f"RACK_SOC == {soc}")

                if viedtd_df.query(f"RACK_SOC == {soc}").empty:
                    closest_row = (viedtd_df['RACK_SOC'] - soc).abs().idxmin()
                    viedtd_soc = viedtd_df.loc[[closest_row]]
                else:
                    viedtd_soc = viedtd_df.query(f"RACK_SOC == {soc}")
                
                init_viectd_v = viectd_soc.loc[viectd_soc.index[0], "RACK_VOLTAGE"]
                init_viedtd_v = viedtd_soc.loc[viedtd_soc.index[0], "RACK_VOLTAGE"]
                
                # n초 뒤의 시간을 계산
                viectd_e_time = viectd_soc.index[0] + timedelta(seconds=time)
                viedtd_e_time = viedtd_soc.index[0] + timedelta(seconds=time)
                
                # 일치하면 일치하는 값을 반환하고 일치 하지 않는다면 가장 가까운 시간
                viectd_nearest_index = viectd_df.index.get_indexer([viectd_e_time], method='nearest')
                viedtd_nearest_index = viedtd_df.index.get_indexer([viedtd_e_time], method='nearest')

                # 데이터 획득
                viectd_nearest_data = viectd_df.iloc[viectd_nearest_index]
                viedtd_nearest_data = viedtd_df.iloc[viedtd_nearest_index]
                viectd = round(abs(viectd_nearest_data["RACK_VOLTAGE"].values[0] - init_viectd_v), 3)
                viedtd = round(abs(viedtd_nearest_data["RACK_VOLTAGE"].values[0] - init_viedtd_v), 3)
                
                print(f"BANK_ID: {bank_id}, RACK_ID: {rack_id}, VIECTD: {viectd}, VIEDTD: {viedtd}")
                result_value[int(bank_id)][int(rack_id)] = {
                        "TIMESTAMP": filename,
                        "VIECTD": viectd,
                        "VIEDTD": viedtd,
                        "S_VIECTD": pd.Timestamp(viectd_soc.index[0]).strftime('%Y-%m-%d %H:%M:%S'),
                        "E_VIECTD": pd.Timestamp(viectd_nearest_data["RACK_VOLTAGE"].index.values[0]).strftime('%Y-%m-%d %H:%M:%S'),
                        "S_VIEDTD": pd.Timestamp(viedtd_soc.index[0]).strftime('%Y-%m-%d %H:%M:%S'),
                        "E_VIEDTD": pd.Timestamp(viedtd_nearest_data["RACK_VOLTAGE"].index.values[0]).strftime('%Y-%m-%d %H:%M:%S'),
                    }
            except ValueError:
                print(f"Error: {bank_id}_{rack_id} no data for VIExTD calculation")
                continue  # 다음 rack_id로 계속 진행
    
    context["task_instance"].xcom_push(key="result", value=result_value)
    print(result_value)



def push_data_to_database(**context):
    result = context["task_instance"].xcom_pull(task_ids="calc_viextd", key="result")
    query_time = context["task_instance"].xcom_pull(task_ids="clac_past_days", key="query_time")

    pg_hook = PostgresHook(postgres_conn_id='ess_stats')
    
    conn = pg_hook.get_conn()
    cur = conn.cursor()

    for bank, rack in result.items():
        bank_id = bank

        for rack_id, value in rack.items():
            try:
                print(f"bank_id = {bank_id}, rack_id = {rack_id}, value = {value}")
                insert_query = """INSERT INTO health_indicator_viextd ("TIMESTAMP", 
                                                                    "OPERATING_SITE", 
                                                                    "BANK_ID", 
                                                                    "RACK_ID", 
                                                                    "S_VIECTD", 
                                                                    "E_VIECTD", 
                                                                    "VIECTD", 
                                                                    "S_VIEDTD", 
                                                                    "E_VIEDTD",
                                                                    "VIEDTD") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                cur.execute(insert_query, (query_time["start_time"], OPERATING_SITE, bank_id, rack_id, value["S_VIECTD"], value["E_VIECTD"], value["VIECTD"], value["S_VIEDTD"], value["E_VIEDTD"], value["VIEDTD"]))
                conn.commit()
            except KeyError:
                print(f"Error: {bank_id}_{rack_id} no data for VIExTD calculation")
                continue # 다음 bank, rack으로 진행

    
    cur.close()
    pg_hook.get_conn().close()

def check_data_for_push(**context):
    result_value = context["task_instance"].xcom_pull(task_ids='calc_viextd', key='result')
    if result_value and any(result_value.values()):
        return 'push_data'
    else:
        return 'end'
    


kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}

with DAG(dag_id='gold_HI_calc_VIExTD',
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

    calc_viextd = PythonOperator(
        task_id='calc_viextd',
        python_callable=calc_health_indicator_viextd
    )

    branch_operator = BranchPythonOperator(
        task_id='branch_check_data',
        python_callable=check_data_for_push,
        provide_context=True,
        dag=dag,
    )

    push_data = PythonOperator(
        task_id='push_data',
        python_callable=push_data_to_database
    )
    complete_data_push = DummyOperator(task_id="complete_data_push")
    end = DummyOperator(task_id="end")

start >> clac_past_time >> calc_viextd >> branch_operator
branch_operator >> push_data >> complete_data_push
branch_operator >> end