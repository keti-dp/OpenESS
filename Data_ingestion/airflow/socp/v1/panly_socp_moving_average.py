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

SAVE_DATA_PATH = "/home/jpark/airflow/dags/calc_socp/dataset/panly/count/ma/"

def clac_past_days(**context):
    execution_date = context["execution_date"]

    seoul_tz = timezone('Asia/Seoul')
    execution_date_seoul = execution_date.astimezone(seoul_tz)

    print(f"Excution_data: {execution_date}")
    print(f"execution_date_seoul: {execution_date_seoul}")
    
    past_5days = (execution_date_seoul - pendulum.duration(days=4)).start_of('day').to_datetime_string()
    past_10days = (execution_date_seoul - pendulum.duration(days=9)).start_of('day').to_datetime_string()
    past_15days = (execution_date_seoul - pendulum.duration(days=14)).start_of('day').to_datetime_string()
    past_30days = (execution_date_seoul - pendulum.duration(days=29)).start_of('day').to_datetime_string()

    end = execution_date_seoul.start_of('day').to_datetime_string()

    query_time = {"past_5days":past_5days, 
                  "past_10days":past_10days,
                  "past_15days":past_15days,
                  "past_30days":past_30days,
                  "end":end}
    print(query_time)
    context["task_instance"].xcom_push(key="query_time", value=query_time)
    

def get_dataset(start, end, period, **context):
    
    f_name = str(end)[:4] + str(end)[5:7] + str(end)[8:10]
    f_list = os.listdir(SAVE_DATA_PATH)
    context["task_instance"].xcom_push(key="f_name", value=f_name)

    if f_name not in f_list:
        try:
            os.mkdir(SAVE_DATA_PATH + f_name)
        except FileExistsError:
            pass

    pg_hook = PostgresHook(postgres_conn_id='ess_stats') 
    query = f"""SELECT * FROM ess_socp_count_oper2 where ("TIMESTAMP" between '{start}' and '{end}')"""
    
    # 쿼리 실행 및 컬럼 정보 가져오기
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    
    # 쿼리 결과를 DataFrame으로 변환
    df = pd.DataFrame(result, columns=column_names)
    print(df)
    df.to_csv(SAVE_DATA_PATH + f_name + "/" + f_name +"_"+ str(period) +".csv", index=False)
    

def calc_moving_average(file, ma_value):
    result_df = pd.DataFrame()
    charge_status_list = [0,1,2] # 충전상태 리스트 : 충전 방전 대기
    soc_range_list = [i for i in range(0, 100, 10)] # 충전상태 리스트 : 충전 방전 대기

    df = pd.read_csv(SAVE_DATA_PATH + file + "/" + file + f"_{ma_value}.csv", index_col="TIMESTAMP", parse_dates=["TIMESTAMP"])
    # 'TIMESTAMP'를 기준으로 오름차순으로 정렬
    df = df.sort_index()

    cell_columns = [col for col in df.columns if 'CELL' in col]

    bank_list = df["BANK_ID"].unique()
    for bank_id in bank_list:
        rack_list = df[df["BANK_ID"] == bank_id]["RACK_ID"].unique()

        for rack_id in rack_list:
            filtered_df = df.query(f"BANK_ID == {bank_id} and RACK_ID == {rack_id}")

            for charge_status in charge_status_list:
                for soc_range in soc_range_list:
                    calc_df = filtered_df.query(f"CHARGE_STATUS == {charge_status} and SOC_RANGE == {soc_range}")

                    df_temp = calc_df[cell_columns].rolling(window=ma_value).mean()
                    df_temp.reset_index(inplace=True)
                    
                    df_temp[['PERIOD', 'BANK_ID', 'RACK_ID', 'CHARGE_STATUS', 'SOC_RANGE']] = calc_df[['PERIOD', 'BANK_ID', 'RACK_ID', 'CHARGE_STATUS', 'SOC_RANGE']].values
                    # Add a column for the MA value
                    df_temp['MA'] = ma_value
                    
                    df_temp = df_temp[df_temp["TIMESTAMP"] == file[:4] + "-" + file[4:6] + "-" +file[6:8]]
                    
                    result_df = pd.concat([result_df, df_temp])
                    
    result_df = result_df.reset_index(drop=True)
    print(result_df)
    result_df.to_csv(SAVE_DATA_PATH + file + "/" + "result" f"_{ma_value}.csv", index=False)
    

def push_data_to_database(table_name, file):
    # PostgresHook을 이용해서 DB에 연결
    pg_hook = PostgresHook(postgres_conn_id='ess_stats')
    conn = pg_hook.get_conn()

    # 데이터프레임 읽기
    f_list = os.listdir(SAVE_DATA_PATH + file + "/")
    result_files = [f for f in f_list if f.startswith('result_')]

    tmp_df = pd.DataFrame()
    for f_name in result_files:
        df = pd.read_csv(SAVE_DATA_PATH + file + "/" + f"{f_name}")
        tmp_df = pd.concat([tmp_df, df])
    tmp_df = tmp_df.reset_index(drop=True)
    print(tmp_df)
    # SQLAlchemy 엔진 생성
    engine = create_engine(pg_hook.get_uri())

    # Session 생성
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # DataFrame을 SQL 데이터베이스로 업로드
        tmp_df.to_sql(table_name, engine, if_exists='append', index=False)
        session.commit()
    finally:
        session.close() # SQLAlchemy session 닫기
        conn.close()    # PostgreSQL connection 닫기



kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}

with DAG(dag_id='panly_socp_moving_average',
         default_args=default_args,
         start_date=datetime(2023, 11, 21, tzinfo=kst),
         schedule_interval=None,
         tags=['panly', 'socp'],
         catchup=True
         ) as dag:

    start = DummyOperator(task_id="start")
    
    clac_past_time = PythonOperator(
        task_id='clac_past_time',
        python_callable=clac_past_days
    )

    get_past_5days_dataset = PythonOperator(
        task_id='get_past_5days_dataset',
        python_callable=get_dataset,
        provide_context=True,
        op_kwargs={
            "start": "{{ task_instance.xcom_pull(task_ids='clac_past_time', key='query_time')['past_5days'] }}",
            "end": "{{ task_instance.xcom_pull(task_ids='clac_past_time', key='query_time')['end'] }}",
            "period": 5
        }
    )

    get_past_10days_dataset = PythonOperator(
        task_id='get_past_10days_dataset',
        python_callable=get_dataset,
        provide_context=True,
        op_kwargs={
            "start": "{{ task_instance.xcom_pull(task_ids='clac_past_time', key='query_time')['past_10days'] }}",
            "end": "{{ task_instance.xcom_pull(task_ids='clac_past_time', key='query_time')['end'] }}",
            "period": 10 
        }
    )

    get_past_15days_dataset = PythonOperator(
        task_id='get_past_15days_dataset',
        python_callable=get_dataset,
        provide_context=True,
        op_kwargs={
            "start": "{{ task_instance.xcom_pull(task_ids='clac_past_time', key='query_time')['past_15days'] }}",
            "end": "{{ task_instance.xcom_pull(task_ids='clac_past_time', key='query_time')['end'] }}",
            "period": 15
        }
    )

    get_past_30days_dataset = PythonOperator(
        task_id='get_past_30days_dataset',
        python_callable=get_dataset,
        provide_context=True,
        op_kwargs={
            "start": "{{ task_instance.xcom_pull(task_ids='clac_past_time', key='query_time')['past_30days'] }}",
            "end": "{{ task_instance.xcom_pull(task_ids='clac_past_time', key='query_time')['end'] }}",
            "period": 30
        }
    )


    calc_ma_5 = PythonOperator(
        task_id='calc_ma_5',
        python_callable=calc_moving_average,
        op_kwargs={
            "file": "{{ task_instance.xcom_pull(task_ids='get_past_5days_dataset', key='f_name')}}",
            "ma_value": 5
        }
    )

    calc_ma_10 = PythonOperator(
        task_id='calc_ma_10',
        python_callable=calc_moving_average,
        op_kwargs={
            "file": "{{ task_instance.xcom_pull(task_ids='get_past_10days_dataset', key='f_name')}}",
            "ma_value": 10
        }
    )

    calc_ma_15 = PythonOperator(
        task_id='calc_ma_15',
        python_callable=calc_moving_average,
        op_kwargs={
            "file": "{{ task_instance.xcom_pull(task_ids='get_past_15days_dataset', key='f_name')}}",
            "ma_value": 15
        }
    )

    calc_ma_30 = PythonOperator(
        task_id='calc_ma_30',
        python_callable=calc_moving_average,
        op_kwargs={
            "file": "{{ task_instance.xcom_pull(task_ids='get_past_30days_dataset', key='f_name')}}",
            "ma_value": 30
        }
    )

    push_data = PythonOperator(
        task_id='push_data',
        python_callable=push_data_to_database,
        op_kwargs={
            "table_name": "ess_socp_count_ma_oper2",
            "file": "{{ task_instance.xcom_pull(task_ids='get_past_5days_dataset', key='f_name')}}"
        }
    )

    end = DummyOperator(task_id="end")

start >> clac_past_time 
clac_past_time >> get_past_5days_dataset >> calc_ma_5 >> push_data >> end
clac_past_time >> get_past_10days_dataset >> calc_ma_10 >> push_data >> end
clac_past_time >> get_past_15days_dataset >> calc_ma_15 >> push_data >> end
clac_past_time >> get_past_30days_dataset >> calc_ma_30 >> push_data >> end