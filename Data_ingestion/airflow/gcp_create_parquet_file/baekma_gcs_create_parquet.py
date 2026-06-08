from datetime import datetime, timedelta
from pytz import  timezone
from pprint import pprint
import pendulum
import pandas as pd
import math
import os
from sqlalchemy import create_engine
from sqlalchemy import JSON
from google.cloud import storage
from google.oauth2 import service_account
import pyarrow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy import DummyOperator

# GCS 버킷 이름 (예: 'ess-bucket-xx')
BUCKET_NAME = ''
CREDENTIALS = service_account.Credentials.from_service_account_file(
    # GCP 서비스 계정 키(JSON) 파일의 절대 경로 (예: '/home/<user>/conf/xxxxx.json')
    ''
)
# DAG 작업 디렉토리의 절대 경로 (예: '/home/<user>/airflow/dags/gcp_create_parquet_file/')
LOCAL_PATH = ""

def tranfer_timezone(**context):
    # UTC 시간인 execution_date 
    execution_date = context["execution_date"]

    seoul_tz = timezone('Asia/Seoul')
    execution_date_seoul = execution_date.astimezone(seoul_tz)

    print(f"Excution_data: {execution_date}")
    print(f"execution_date_seoul: {execution_date_seoul}")
    start_time = execution_date_seoul.start_of('day').to_datetime_string()
    
    date = {"date_time":start_time}
    print(date)

    context["task_instance"].xcom_push(key="date", value=date)


def convert_feather_to_parquet(**context):
    date = context["task_instance"].xcom_pull(task_ids="transfer_tz", key="date")
    path = date["date_time"][:4] + "/" + date["date_time"][5:7] + "/" + date["date_time"][8:10]

    # GCS 클라이언트 초기화
    client = storage.Client(credentials=CREDENTIALS)
    
    # 버킷에서 파일 목록 가져오기
    bucket = client.get_bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=path)

    for blob in blobs:
        # # Feather 파일 경로 설정
        feather_file_path = blob.name

        # 파일 이름만 추출 ex) 입력: /path/to/some/file.txt, 결과: file.txt 
        filename = os.path.basename(feather_file_path)
        filename_base  = filename.split(".")[0]
        
        # parquet 파일이 있으면 해당 작업 건너뛰기
        if filename.endswith(('.parquet')):
            continue

        if not filename.endswith(('.ft', '.feather')):
            continue
        
        print(f"{filename} 변환 중입니다. ")
        
        # Path만 추출 ex) 입력: /path/to/some/file.txt, 결과: /path/to/some/ 
        gcs_path = os.path.dirname(feather_file_path)
        
        try:
            # Feather 파일을 DataFrame으로 읽기
            blob.download_to_filename(LOCAL_PATH + filename)
            df = pd.read_feather(LOCAL_PATH + filename)

            # DataFrame을 Parquet 파일로 저장
            parquet_filename = filename_base + ".parquet"
            df.to_parquet(LOCAL_PATH + parquet_filename)

            # Parquet 파일을 GCS에 업로드
            parquet_blob = bucket.blob(os.path.join(gcs_path, parquet_filename))
            parquet_blob.upload_from_filename(LOCAL_PATH + parquet_filename)
            
        except pyarrow.lib.ArrowInvalid as e:
            print(f"오류 발생: {e}. 파일: {filename}은(는) 변환할 수 없습니다. 다음 파일로 넘어갑니다.")
            continue  # 다음 파일로 넘어가기

        finally:
            # 로컬에 저장된 임시 파일 삭제
            if os.path.exists(LOCAL_PATH + filename):
                os.remove(LOCAL_PATH + filename)
            if os.path.exists(LOCAL_PATH + parquet_filename):
                os.remove(LOCAL_PATH + parquet_filename)

kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}



with DAG(dag_id='beakma_gcs_create_parquet',
         default_args=default_args,
         start_date=datetime(2024, 4, 17, tzinfo=kst),
         schedule_interval='40 00 * * *',
         tags=['baekma', 'dataset'],
         catchup=True
         ) as dag:

    start_task = DummyOperator(
        task_id="start_task"
    )
    
    transfer_tz = PythonOperator(
        task_id='transfer_tz',
        python_callable=tranfer_timezone
    )

    convert_file = PythonOperator(
        task_id='convert_file',
        python_callable=convert_feather_to_parquet
    )

    end_task = DummyOperator(
        task_id='end_task'
    )

start_task >> transfer_tz >> convert_file >> end_task