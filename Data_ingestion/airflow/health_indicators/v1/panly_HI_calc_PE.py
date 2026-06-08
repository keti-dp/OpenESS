from datetime import datetime, timedelta
from pytz import  timezone
from pprint import pprint
import pandas as pd
import os
import json
import copy
import pendulum
import numpy as np

from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import BranchPythonOperator

DATA_SAVE_PATH = "/home/jpark/airflow/dags/calc_health_indicator/dataset/panly/"
DIFF_SOC = 3
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
    

def calc_health_indicator_partial_energy(**context):
    dag_conf = context["dag_run"].conf
    filename = dag_conf.get("filename")
    
    df = pd.read_feather(DATA_SAVE_PATH + filename + ".feather")

    result_value = {}
    s_time, e_time, v_min, i_min, v_max, i_max, soc_min, soc_max, pe = (None, )*9

    soc = 60
    for bank_id in list(df["BANK_ID"].unique()):
        result_value[int(bank_id)] = {}  # bank_id에 대한 내부 딕셔너리 생성
        bank_df = df.query(f"BANK_ID == {bank_id}")

        for rack_id in list(bank_df["RACK_ID"].unique()):          
            # rack_id에 대한 내부 딕셔너리 생성
            result_value[int(bank_id)][int(rack_id)] = {}  

            rack_pe_df = bank_df.query(f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_CHARGE == 1").reset_index(drop=True)

            # 'TIMESTAMP'를 index로 설정 및 중복제거
            rack_pe_df = rack_pe_df.set_index('TIMESTAMP')    
            rack_pe_df = rack_pe_df.loc[~rack_pe_df.index.duplicated(keep='first')]
            
            # SOC 60%인 데이터, 없다면 근처 데이터를 쿼리
            try:
                if rack_pe_df.query(f"RACK_SOC == {soc}").empty:
                    closest_row = (rack_pe_df['RACK_SOC'] - soc).abs().idxmin()
                    pe_soc = rack_pe_df.loc[[closest_row]]
                else:
                    pe_soc = rack_pe_df.query(f"RACK_SOC == {soc}")
            except ValueError:
                print(f"Error: {bank_id}_{rack_id} no data for PE calculation")
                continue
            saved_soc = pe_soc["RACK_SOC"].unique()[0]

            # soc 60 or soc 60이 없을 시 존재하는 가장 근처의 값 일때의 v의 최소 값
            vmin_at_soc_60 = pe_soc["RACK_VOLTAGE"].min()

            # SOC 60일(60이 없을시 근처 값)에서부터 전압에 10V를 더한 데이터 쿼리
            v_plus_10 = vmin_at_soc_60 + 10
            pe_df = rack_pe_df.query(f"RACK_SOC >= {saved_soc} & RACK_VOLTAGE >= {vmin_at_soc_60} & RACK_VOLTAGE <= {v_plus_10}")
            
            # SOC 조건이 너무 차이나면 패스
            if abs(saved_soc-soc) > DIFF_SOC:
                print(f"BANK_ID: {bank_id}, RACK_ID: {rack_id}의 참조된 SOC값: {saved_soc}")
                continue
            
            # index.min과 index.max 사이에 데이터가 없을 경우 이전 값으로 채우기
            min_time = pe_df.index.min()
            max_time = pe_df.index.max()
            time_range = pd.date_range(start=min_time, end=max_time, freq='S')  # 'S'는 초를 의미

            pe_df = pe_df.reindex(time_range).fillna(method='ffill')

            # 각 시간 단계에서의 전압과 전류
            voltages = pe_df["RACK_VOLTAGE"].to_numpy()
            currents = pe_df["RACK_CURRENT"].to_numpy()
            # 시간 간격 (예를 들어, 데이터가 1초마다 샘플링된 경우)
            delta_t = 1  # 1 second (이 값을 실제 데이터에 맞게 설정해야 함)
            # 부분 에너지 계산
            partial_energy = round(np.sum(voltages * currents * delta_t) / 3600, 4)

            s_time = pe_df.index.min().strftime('%Y-%m-%d %H:%M:%S')
            e_time = pe_df.index.max().strftime('%Y-%m-%d %H:%M:%S')
            v_min = pe_df["RACK_VOLTAGE"].min()
            i_min = pe_df["RACK_CURRENT"].min()
            v_max = pe_df["RACK_VOLTAGE"].max()
            i_max = pe_df["RACK_CURRENT"].max()
            soc_min = pe_df["RACK_SOC"].min()
            soc_max = pe_df["RACK_SOC"].max()
            pe = partial_energy
        
            print(f"==========================BANK_ID: {bank_id}, RACK_ID: {rack_id}==========================")
            print(f"Partial Energy 구간의 시작 시간: {s_time}, 최대 SOC: {e_time}")
            print(f"Partial Energy 구간의 최소 SOC: {soc_min}, 최대 SOC: {soc_max}")
            print(f"Partial Energy 구간의 최소 전압: {v_min}, 최대 전압: {v_max}")
            print(f"Partial Energy 구간의 최소 전류: {i_min}, 최대 전류: {i_max}")
            print(f"Partial Energy: {pe}")

            result_value[int(bank_id)][int(rack_id)] = {
                    "S_TIME": s_time,
                    "E_TIME": e_time,
                    "MIN_SOC": soc_min,
                    "MAX_SOC": soc_max,
                    "MIN_VOLTAGE": v_min,
                    "MAX_VOLTAGE": v_max,
                    "MIN_CURRENT": i_min,
                    "MAX_CURRENT": i_max,
                    "PE": pe
                }
    
    context["task_instance"].xcom_push(key="result", value=result_value)
    pprint(result_value)

def check_data_for_push(**context):
    result_value = context["task_instance"].xcom_pull(task_ids='calc_partial_energy', key='result')
    if result_value and any(result_value.values()):
        return 'push_data'
    else:
        return 'end'

def push_data_to_database(**context):
    result = context["task_instance"].xcom_pull(task_ids="calc_partial_energy", key="result")
    query_time = context["task_instance"].xcom_pull(task_ids="clac_past_days", key="query_time")

    pg_hook = PostgresHook(postgres_conn_id='ess_stats')
    
    conn = pg_hook.get_conn()
    cur = conn.cursor()

    for bank, rack in result.items():
        bank_id = bank

        for rack_id, value in rack.items():
            print(f"bank_id = {bank_id}, rack_id = {rack_id}, value = {value}")
            if not value:
                continue
            else:
                insert_query = """INSERT INTO health_indicator_pe ("TIMESTAMP", 
                                                                    "OPERATING_SITE", 
                                                                    "BANK_ID", 
                                                                    "RACK_ID", 
                                                                    "S_TIME",
                                                                    "E_TIME",
                                                                    "MIN_SOC",
                                                                    "MAX_SOC",
                                                                    "MIN_VOLTAGE",
                                                                    "MAX_VOLTAGE",
                                                                    "MIN_CURRENT",
                                                                    "MAX_CURRENT",
                                                                    "PE"
                                                                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                cur.execute(insert_query, (query_time["start_time"], OPERATING_SITE, bank_id, rack_id, value["S_TIME"], value["E_TIME"], value["MIN_SOC"], value["MAX_SOC"], value["MIN_VOLTAGE"], value["MAX_VOLTAGE"], value["MIN_CURRENT"], value["MAX_CURRENT"], value["PE"]))
                conn.commit()
    
    cur.close()
    pg_hook.get_conn().close()



kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}

with DAG(dag_id='panly_HI_calc_PE',
         default_args=default_args,
         start_date=datetime(2023, 11, 22, tzinfo=kst),
         schedule_interval=None,
         tags=['panly', 'indicator', 'rack'],
         catchup=True
         ) as dag:

    start = DummyOperator(task_id="start")
    
    clac_past_time = PythonOperator(
        task_id='clac_past_days',
        python_callable=clac_past_days
    )

    calc_partial_energy = PythonOperator(
        task_id='calc_partial_energy',
        python_callable=calc_health_indicator_partial_energy
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


    start >> clac_past_time >> calc_partial_energy >> branch_operator
    branch_operator >> push_data >> complete_data_push
    branch_operator >> end