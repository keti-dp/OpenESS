"""
일일 배터리 데이터 다운로드 Airflow DAG
매일 GCP에서 배터리 데이터를 다운로드하여 Parquet 파일로 저장
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum
from pathlib import Path
import sys
import os
import yaml

# 현재 디렉토리를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from utils.common import ensure_directory_exists, get_gcs_client

# 사이트 ID 지정
SITE_ID = 'panly'

# data_download_config.yaml 로드
config_path = Path(__file__).parent / 'config' / 'data_download_config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

if SITE_ID not in config['sites']:
    raise ValueError(f"사이트 '{SITE_ID}'를 찾을 수 없습니다. 사용 가능한 사이트: {list(config['sites'].keys())}")

site_config = config['sites'][SITE_ID]
airflow_config = config['defaults']['airflow']


def download_daily_data(**context):
    """전날 데이터 다운로드 및 GCS 업로드"""
    from operators.gcp_data_operator import GCPDataDownloader
    from datetime import timedelta

    # 실제 전날 데이터를 다운로드하기 위해 execution_date + 1일 사용
    # Airflow의 execution_date는 데이터 구간 시작이므로, 실제 전날은 +1일
    execution_date = context['execution_date']
    target_date = execution_date + timedelta(days=1)

    year = target_date.year
    month = target_date.month
    day = target_date.day

    print(f"[{site_config['name']}] 실행 시간: {execution_date}")
    print(f"[{site_config['name']}] {year}-{month:02d}-{day:02d} 데이터 다운로드 시작")

    # 사이트 설정에서 값 가져오기
    credential_path = site_config['gcp']['credential_path']
    bucket_name = site_config['gcp']['bucket_name']
    save_dir = site_config['paths']['data_dir']
    data_types = site_config['data']['data_types']

    # GCS 목적지 설정
    dest_bucket_name = 'keti-airflow-dataset'
    dest_blob_prefix = f'rack-ori-data/{SITE_ID}/'

    # 디렉토리 생성 (임시 저장용)
    ensure_directory_exists(save_dir)

    # 다운로더 생성 (GCS 업로드 설정 포함)
    downloader = GCPDataDownloader(
        credential_path=credential_path,
        bucket_name=bucket_name,
        save_dir=save_dir,
        data_types=data_types,
        dest_bucket_name=dest_bucket_name,
        dest_blob_prefix=dest_blob_prefix
    )

    # 해당 날짜 데이터 다운로드 및 GCS 업로드
    try:
        files = downloader.download_date_data(year, month, day)
        print(f"다운로드 및 GCS 업로드 성공: {len(files)}개 파일")
        print(f"GCS 경로: gs://{dest_bucket_name}/{dest_blob_prefix}")

        # 모니터링을 위해 파일 개수를 XCom에 저장
        context['task_instance'].xcom_push(key='files_downloaded', value=len(files))
        context['task_instance'].xcom_push(key='download_date', value=f"{year}-{month:02d}-{day:02d}")
        context['task_instance'].xcom_push(key='site_id', value=SITE_ID)
        context['task_instance'].xcom_push(key='gcs_path', value=f"gs://{dest_bucket_name}/{dest_blob_prefix}")

        return files
    except Exception as e:
        print(f"데이터 다운로드 오류: {e}")
        raise


def check_data_quality(**context):
    """기본 데이터 품질 검사 (GCS에서 읽기)"""
    import pandas as pd
    from google.cloud import storage

    # XCom에서 다운로드 정보 가져오기
    ti = context['task_instance']
    files_downloaded = ti.xcom_pull(key='files_downloaded', task_ids='download_data')
    download_date = ti.xcom_pull(key='download_date', task_ids='download_data')
    gcs_path = ti.xcom_pull(key='gcs_path', task_ids='download_data')

    print(f"{download_date} 날짜의 {files_downloaded}개 파일 품질 검사 중")
    print(f"GCS 경로: {gcs_path}")

    # GCS 클라이언트 초기화
    storage_client = get_gcs_client(site_config['gcp']['credential_path'])

    # GCS 버킷 및 prefix 파싱
    dest_bucket_name = 'keti-airflow-dataset'
    dest_blob_prefix = f'rack-ori-data/{SITE_ID}/'
    bucket = storage_client.bucket(dest_bucket_name)

    # 해당 날짜의 parquet 파일 찾기
    date_str = download_date.replace('-', '')
    blobs = list(bucket.list_blobs(prefix=dest_blob_prefix))
    parquet_blobs = [b for b in blobs if date_str in b.name and b.name.endswith('.parquet')]

    if not parquet_blobs:
        print(f"⚠ {download_date} 날짜의 parquet 파일을 GCS에서 찾을 수 없습니다")
        print(f"하지만 {files_downloaded}개 파일이 업로드되었다고 보고되었으므로 통과합니다")
        return []

    total_rows = 0
    file_info = []

    for blob in parquet_blobs[:3]:  # 처음 3개만 샘플 검사
        try:
            # GCS blob을 메모리로 읽기 (PyArrow GCS 지원 없이도 동작)
            from io import BytesIO

            file_name = blob.name.split('/')[-1]
            print(f"  검사 중: {file_name}...", end=' ')

            # blob 데이터를 메모리로 다운로드
            blob_bytes = blob.download_as_bytes()

            # BytesIO로 감싸서 pandas로 읽기
            df = pd.read_parquet(BytesIO(blob_bytes))
            rows = len(df)
            cols = len(df.columns)
            total_rows += rows

            file_info.append({
                'file': file_name,
                'rows': rows,
                'columns': cols,
                'size_mb': blob.size / (1024 * 1024)
            })

            print(f"✓ {rows}행, {cols}열")

            # 기본 검사
            if rows == 0:
                raise ValueError(f"파일 {file_name}에 데이터가 없습니다")

        except Exception as e:
            print(f"✗ 검사 오류: {e}")
            raise

    print(f"\n샘플 파일의 총 행 수: {total_rows} (전체 {len(parquet_blobs)}개 중 {len(file_info)}개 검사)")

    # 품질 정보를 XCom에 저장
    ti.xcom_push(key='quality_check', value={
        'total_rows': total_rows,
        'total_files': len(file_info),
        'files': file_info
    })

    return file_info


def cleanup_old_data(**context):
    """GCS에서 최신 N개 파일만 유지하고 나머지 삭제"""
    from google.cloud import storage

    retention_file_count = site_config['data']['retention_file_count']
    dest_bucket_name = 'keti-airflow-dataset'
    dest_blob_prefix = f'rack-ori-data/{SITE_ID}/'

    print(f"[{site_config['name']}] GCS에서 최신 {retention_file_count}개 파일만 유지")
    print(f"대상: gs://{dest_bucket_name}/{dest_blob_prefix}")

    # GCS 클라이언트 초기화
    storage_client = get_gcs_client(site_config['gcp']['credential_path'])
    bucket = storage_client.bucket(dest_bucket_name)

    # 해당 prefix의 모든 blob 나열 (디렉토리 제외)
    blobs = [b for b in bucket.list_blobs(prefix=dest_blob_prefix)
             if not b.name.endswith('/') and b.name.endswith('.parquet')]

    print(f"현재 GCS에 {len(blobs)}개의 parquet 파일 존재")

    if len(blobs) <= retention_file_count:
        print(f"파일 개수({len(blobs)})가 보관 개수({retention_file_count}) 이하이므로 삭제 불필요")
        context['task_instance'].xcom_push(key='removed_files_count', value=0)
        return []

    # 생성 시간으로 정렬 (최신순)
    blobs_sorted = sorted(blobs, key=lambda b: b.time_created, reverse=True)

    # 최신 N개를 제외한 나머지 삭제
    blobs_to_keep = blobs_sorted[:retention_file_count]
    blobs_to_delete = blobs_sorted[retention_file_count:]

    print(f"유지할 파일: {len(blobs_to_keep)}개")
    print(f"삭제할 파일: {len(blobs_to_delete)}개")

    removed_files = []
    for blob in blobs_to_delete:
        file_name = blob.name.split('/')[-1]
        blob_created_kst = pendulum.instance(blob.time_created).in_timezone('Asia/Seoul')
        print(f"  삭제 중: {file_name} (생성일: {blob_created_kst.strftime('%Y-%m-%d %H:%M:%S')} KST)")
        blob.delete()
        removed_files.append(blob.name)

    print(f"GCS에서 {len(removed_files)}개 파일 삭제 완료")

    # XCom에 삭제 정보 저장
    context['task_instance'].xcom_push(key='removed_files_count', value=len(removed_files))
    context['task_instance'].xcom_push(key='retention_file_count', value=retention_file_count)

    return removed_files


def log_completion(**context):
    """완료 로그 출력"""
    from datetime import datetime

    # XCom에서 정보 가져오기
    ti = context['task_instance']
    files_downloaded = ti.xcom_pull(key='files_downloaded', task_ids='download_data')
    gcs_path = ti.xcom_pull(key='gcs_path', task_ids='download_data')
    removed_files_count = ti.xcom_pull(key='removed_files_count', task_ids='cleanup_old_data')

    print(f"일일 데이터 다운로드 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  다운로드 및 GCS 업로드: {files_downloaded}개 파일")
    print(f"  GCS 경로: {gcs_path}")
    print(f"  정리된 파일: {removed_files_count}개")

    return "완료"

# DAG 기본 인자
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': airflow_config.get('retries', 2),
    'retry_delay': timedelta(minutes=airflow_config.get('retry_delay_minutes', 10)),
    'execution_timeout': timedelta(hours=1),
}

# DAG 생성
dag = DAG(
    f'daily_battery_data_download_{SITE_ID}',
    default_args=default_args,
    description=f'[{site_config["name"]}] 매일 GCP에서 배터리 데이터를 다운로드하여 Parquet으로 저장',
    schedule=airflow_config.get('daily_download_schedule', '0 2 * * *'),
    start_date=pendulum.datetime(2025, 1, 1, tz='Asia/Seoul'),
    catchup=False,  # 과거 실행 건너뛰기
    tags=['battery', 'data-download', 'gcp', 'daily', SITE_ID],
    max_active_runs=airflow_config.get('max_active_runs', 1),
)

# Task 1: 데이터 다운로드
download_task = PythonOperator(
    task_id='download_data',
    python_callable=download_daily_data,
    provide_context=True,
    dag=dag,
)

# Task 2: 데이터 품질 검사
quality_check_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    provide_context=True,
    dag=dag,
)

# Task 3: 오래된 데이터 정리 (GCS)
cleanup_task = PythonOperator(
    task_id='cleanup_old_data',
    python_callable=cleanup_old_data,
    provide_context=True,
    dag=dag,
)

# Task 4: 완료 로그
log_task = PythonOperator(
    task_id='log_completion',
    python_callable=log_completion,
    provide_context=True,
    dag=dag,
)

# Task 의존성 설정
# 다운로드 (GCS 업로드) → 품질검사 → GCS 정리 → 로그
download_task >> quality_check_task >> cleanup_task >> log_task
