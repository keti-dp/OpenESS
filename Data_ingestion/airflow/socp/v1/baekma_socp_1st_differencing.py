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

SAVE_DATA_PATH = "/home/jpark/airflow/dags/calc_socp/dataset/baekma/count/diff/"

def calc_past_days(**context):
    execution_date = context["execution_date"]

    seoul_tz = timezone('Asia/Seoul')
    execution_date_seoul = execution_date.astimezone(seoul_tz)

    print(f"Excution_data: {execution_date}")
    print(f"execution_date_seoul: {execution_date_seoul}")
      
    start = (execution_date_seoul - timedelta(days=1)).start_of('day').to_datetime_string()
    end = execution_date_seoul.start_of('day').to_datetime_string()

    query_time = {"start":start, "end":end}
    print(query_time)
    context["task_instance"].xcom_push(key="query_time", value=query_time)
    

def get_dataset(start, end, **context):
    
    f_name = str(end)[:4] + str(end)[5:7] + str(end)[8:10]
    f_list = os.listdir(SAVE_DATA_PATH)
    context["task_instance"].xcom_push(key="f_name", value=f_name)
    print("==================")
    print(f_name)
    print("==================")
    if f_name not in f_list:
        os.mkdir(SAVE_DATA_PATH + f_name)

    pg_hook = PostgresHook(postgres_conn_id='ess_stats') 
    query = f"""SELECT * FROM ess_socp_count_oper4 where ("TIMESTAMP" between '{start}' and '{end}')"""
    #result = pg_hook.get_records(query)
    
    # 쿼리 실행 및 컬럼 정보 가져오기
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    
    # 쿼리 결과를 DataFrame으로 변환
    df = pd.DataFrame(result, columns=column_names)
    print(df)
    df.to_csv(SAVE_DATA_PATH + f_name + "/" + f_name +".csv", index=False)
    


def calc_1st_differencing(file):
    result_df = pd.DataFrame()
    charge_status_list = [0,1,2] # 충전상태 리스트 : 충전 방전 대기
    soc_range_list = [i for i in range(0, 100, 10)] # 충전상태 리스트 : 충전 방전 대기

    df = pd.read_csv(SAVE_DATA_PATH + file + "/" + file + ".csv", index_col="TIMESTAMP", parse_dates=["TIMESTAMP"])
    # 'TIMESTAMP'를 기준으로 오름차순으로 정렬
    df = df.sort_index()
    
    bank_list = df["BANK_ID"].unique()
    for bank_id in bank_list:
        rack_list = df[df["BANK_ID"] == bank_id]["RACK_ID"].unique()

        for rack_id in rack_list:
            filtered_df = df.query(f"BANK_ID == {bank_id} and RACK_ID == {rack_id}")

            for charge_status in charge_status_list:
                for soc_range in soc_range_list:
                    calc_df = filtered_df.query(f"CHARGE_STATUS == {charge_status} and SOC_RANGE == {soc_range}")
                    
                    # diff 결과를 저장할 dict
                    diff_data = {}

                    # 원래 DataFrame의 복사본을 만들고 그 위에서 작업을 수행합니다.
                    
                    for i in range(1, 241):
                        cell_col = 'CELL_' + str(i)
                        calc_df_copy = calc_df.copy()
                        
                        # 'shift' 함수를 사용하여 이전 날짜의 값을 얻습니다.
                        calc_df_copy['PREV_' + cell_col] = calc_df_copy[cell_col].shift(1)
                        # 'diff' 함수를 사용하여 이전날과 다음날의 차이를 계산하고, 결과를 딕셔너리에 추가합니다.
                        diff_data['DIFF_' + cell_col] = calc_df_copy[cell_col] - calc_df_copy['PREV_' + cell_col]

                    # 기존 데이터프레임에서 필요한 열들을 선택하여 새로운 데이터프레임에 추가합니다.
                    preserve_cols = ['PERIOD', 'BANK_ID', 'RACK_ID', 'CHARGE_STATUS', 'SOC_RANGE']
                    preserve_df = calc_df_copy[preserve_cols].copy()
                    
                    # 차이를 계산한 데이터와 원래의 필요한 열들을 합칩니다.
                    diff_df = pd.concat([preserve_df, pd.DataFrame(diff_data)], axis=1)
                    diff_df.index.name = 'TIMESTAMP'

                    # 인덱스를 컬럼으로 변환하는 경우
                    diff_df.reset_index(inplace=True)
                    
                    diff_df = diff_df[diff_df["TIMESTAMP"] == file[:4] + "-" + file[4:6] + "-" +file[6:8]]
                    
                    result_df = pd.concat([result_df, diff_df])
    print(result_df)
    result_df.to_csv(SAVE_DATA_PATH + file + "/" + "result.csv", index=False)




def push_data_to_database(table_name, file):        
    # PostgresHook을 이용해서 DB에 연결
    pg_hook = PostgresHook(postgres_conn_id='ess_stats')
    conn = pg_hook.get_conn()

    # 데이터프레임 읽기
    df = pd.read_csv(SAVE_DATA_PATH + file + "/" + "result.csv")
    print(df)

    # SQLAlchemy 엔진 생성
    engine = create_engine(pg_hook.get_uri())

    # Session 생성
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # DataFrame을 SQL 데이터베이스로 업로드
        df.to_sql(table_name, engine, if_exists='append', index=False)
        session.commit()
    finally:
        session.close() # SQLAlchemy session 닫기
        conn.close()    # PostgreSQL connection 닫기



kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}

with DAG(dag_id='baekma_SoCP_1st_differencing',
         default_args=default_args,
         start_date=datetime(2024, 1, 1, tzinfo=kst),
         schedule_interval=None,
         tags=['baekma', 'socp'],
         catchup=True
         ) as dag:

    start = DummyOperator(task_id="start")
    
    calc_past_time = PythonOperator(
        task_id='calc_past_time',
        python_callable=calc_past_days
    )

    getDataset = PythonOperator(
        task_id='getDataset',
        python_callable=get_dataset,
        provide_context=True,
        op_kwargs={
            "start": "{{ task_instance.xcom_pull(task_ids='calc_past_time', key='query_time')['start'] }}",
            "end": "{{ task_instance.xcom_pull(task_ids='calc_past_time', key='query_time')['end'] }}",
        }
    )

    calc_1st_diff = PythonOperator(
        task_id='calc_1st_diff',
        python_callable=calc_1st_differencing,
        op_kwargs={
            "file": "{{ task_instance.xcom_pull(task_ids='getDataset', key='f_name')}}",
        }
    )

    push_data = PythonOperator(
        task_id='push_data',
        python_callable=push_data_to_database,
        op_kwargs={
            "table_name": "ess_socp_count_diff_oper4",
            "file": "{{ task_instance.xcom_pull(task_ids='getDataset', key='f_name')}}"
        }
    )

    end = DummyOperator(task_id="end")

start >> calc_past_time >> getDataset >> calc_1st_diff >> push_data >> end






