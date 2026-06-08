from datetime import datetime, timedelta
from pytz import  timezone
from pprint import pprint
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import pandas as pd
import os
import pendulum

from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy import DummyOperator

SAVE_DATA_PATH = "/home/jpark/airflow/dags/MaxVol_multistep_forecasting/dataset/panly/"
COL_LIST = ['RACK_MAX_CELL_VOLTAGE', 'RACK_VOLTAGE',
            'RACK_MAX_CELL_TEMPERATURE','RACK_CURRENT',
            'RACK_SOC', "RACK_ID", "TIMESTAMP"]

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


def get_dataset(**context):
    query_time = context["task_instance"].xcom_pull(task_ids="calc_past_time", key="query_time")
    filename = str(query_time["start_time"])[:10]

    queries = f"""select
                    "RACK_MAX_CELL_VOLTAGE",
                    "RACK_VOLTAGE",
                    "RACK_MAX_CELL_TEMPERATURE",
                    "RACK_CURRENT",
                    "RACK_SOC",
                    "BANK_ID",
                    "RACK_ID",
                    "TIMESTAMP"
                  from 
                    rack 
                  where 
                    "TIMESTAMP" between '{query_time["start_time"]}' and '{query_time["end_time"]}'"""
    pg_hook = PostgresHook(postgres_conn_id='panly_site')
    df = pg_hook.get_pandas_df(sql=queries)
    df["TIMESTAMP"] = df["TIMESTAMP"].dt.tz_convert('Asia/Seoul')

    #누락 데이터 생성
    concat_df = pd.DataFrame()
    for bank_id in list(df["BANK_ID"].unique()):
        select_df = df.query(f"""BANK_ID == {bank_id}""")

        for rack_id in list(select_df["RACK_ID"].unique()):
            filtered_df = select_df.query(f"""RACK_ID == {rack_id}""")
            
            # 중복 제거
            filtered_df = filtered_df.drop_duplicates(subset='TIMESTAMP', keep='first')

            # 누락데이터 생성
            filtered_df['TIMESTAMP'] = pd.to_datetime(filtered_df['TIMESTAMP'])
            filtered_df.set_index('TIMESTAMP', inplace=True)

            start_time = filtered_df.index.min().replace(hour=0, minute=0, second=0)
            end_time = filtered_df.index.max().replace(hour=23, minute=59, second=59)

            full_range = pd.date_range(start=start_time, end=end_time, freq='S')  # 초단위
            
            
            
            filtered_df = filtered_df.reindex(full_range).fillna(method='bfill').fillna(method='ffill')
            filtered_df["minute"] = filtered_df.index.minute
            filtered_df["second"] = filtered_df.index.second
            
            concat_df = pd.concat([concat_df, filtered_df])
    
    concat_df.reset_index(inplace=True)
    concat_df.rename(columns={'index': 'TIMESTAMP'}, inplace=True)
    concat_df = concat_df.sort_values(by=['TIMESTAMP', 'BANK_ID', 'RACK_ID'] ,ascending=True).reset_index(drop=True)
    print(concat_df)

    concat_df.to_feather(SAVE_DATA_PATH + filename + ".feather")


kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}

with DAG(dag_id='panly_get_1day_forecasting_dataset',
         default_args=default_args,
         start_date=datetime(2023, 1, 1, tzinfo=kst),
         schedule_interval='00 01 * * *',
         tags=['forecasting', 'dataset'],
         catchup=True
         ) as dag:
    
    start = DummyOperator(task_id="start")

    calc_past_time = PythonOperator(
        task_id='calc_past_time',
        python_callable=clac_past_days
    )

    download_dataset = PythonOperator(
        task_id='get_dataset',
        python_callable=get_dataset
    )

    end = DummyOperator(task_id="end")


start >> calc_past_time >> download_dataset >> end
