"""여러 DAG에서 사용되는 공통 operator 함수들"""

import sys
from pathlib import Path

# DAG 파일의 디렉토리를 Python 경로에 추가
DAG_DIR = Path(__file__).parent.parent.resolve()
if str(DAG_DIR) not in sys.path:
    sys.path.insert(0, str(DAG_DIR))

import os
import shutil
from datetime import datetime
from pytz import timezone
from typing import Dict, Any

from airflow.providers.postgres.hooks.postgres import PostgresHook
from config.site_config import SiteConfig


def initialize_global_variables(site_config: SiteConfig, **context) -> None:
    """전역 변수를 초기화하고 XCom에 push합니다.

    Args:
        site_config: 사이트 설정 객체
        **context: Airflow context
    """
    print(f"Initialized DATA_SAVE_PATH: {site_config.data_path}")
    context['task_instance'].xcom_push(key='DATA_SAVE_PATH', value=site_config.data_path)
    context['task_instance'].xcom_push(key='OPERATING_SITE', value=site_config.site_id)


def calculate_time_range(**context) -> None:
    """데이터 쿼리를 위한 시간 범위를 계산하고 XCom에 push합니다.

    Args:
        **context: Airflow context
    """
    execution_date = context["execution_date"]
    seoul_tz = timezone('Asia/Seoul')
    execution_date_seoul = execution_date.astimezone(seoul_tz)

    print(f"Execution_date: {execution_date}")
    print(f"execution_date_seoul: {execution_date_seoul}")

    start_time = execution_date_seoul.start_of('day').to_datetime_string()
    end_time = execution_date_seoul.end_of('day').to_datetime_string()

    query_time = {"start_time": start_time, "end_time": end_time}
    print(f"Query time range: {query_time}")
    context["task_instance"].xcom_push(key="query_time", value=query_time)


def execute_select_query(site_config: SiteConfig, **context) -> None:
    """SELECT 쿼리를 실행하여 데이터셋을 가져오고 parquet로 저장합니다.

    Args:
        site_config: 사이트 설정 객체
        **context: Airflow context
    """
    import pandas as pd

    data_save_path = context["task_instance"].xcom_pull(task_ids="initialize_globals", key="DATA_SAVE_PATH")
    query_time = context["task_instance"].xcom_pull(task_ids="calc_time_range", key="query_time")

    # 사이트의 배터리 상태 필드를 기반으로 쿼리 생성
    status_fields = site_config.battery_status_fields
    status_select = ",\n                ".join([f'bk."{field}"' for field in status_fields])
    status_columns = ",\n                    ".join([f'"{field}"' for field in status_fields])

    queries = f"""
        SELECT DISTINCT
            rk."TIMESTAMP",
            rk."BANK_ID",
            rk."RACK_ID",
            rk."RACK_SOC",
            rk."RACK_CURRENT",
            rk."RACK_VOLTAGE",
            rk."RACK_MAX_CELL_VOLTAGE",
            rk."RACK_MIN_CELL_VOLTAGE",
            {status_select}
        FROM (
            SELECT
                "TIMESTAMP",
                "BANK_ID",
                "RACK_ID",
                "RACK_SOC",
                "RACK_CURRENT",
                "RACK_VOLTAGE",
                "RACK_MAX_CELL_VOLTAGE",
                "RACK_MIN_CELL_VOLTAGE"
            FROM rack
            WHERE
                ("TIMESTAMP" BETWEEN '{query_time["start_time"]}' AND '{query_time["end_time"]}')
        ) AS rk
        INNER JOIN (
            SELECT
                "TIMESTAMP",
                {status_columns}
            FROM bank
            WHERE
                ("TIMESTAMP" BETWEEN '{query_time["start_time"]}' AND '{query_time["end_time"]}')
        ) AS bk ON rk."TIMESTAMP" = bk."TIMESTAMP"
        ORDER BY "TIMESTAMP" DESC;
    """

    pg_hook = PostgresHook(postgres_conn_id=site_config.postgres_conn_id)
    df = pg_hook.get_pandas_df(sql=queries)
    df = df.sort_values(by=['TIMESTAMP', 'BANK_ID', 'RACK_ID'], ascending=True).reset_index(drop=True)

    df["TIMESTAMP"] = df["TIMESTAMP"].dt.tz_convert('Asia/Seoul')
    filename = str(query_time["start_time"])[:10]

    context["task_instance"].xcom_push(key="filename", value=filename)
    df.to_parquet(data_save_path + filename + ".parquet")

    print(f"Dataset saved: {filename}.parquet")
    print(f"Rows: {len(df)}")


def check_data_for_push(task_id: str, expected_keys: list = None, **context) -> str:
    """데이터베이스에 push할 데이터가 있는지 확인합니다.

    Args:
        task_id: 결과를 가져올 Task ID
        expected_keys: 결과 dict에서 기대되는 키 목록 (선택사항)
        **context: Airflow context

    Returns:
        데이터가 있으면 'push_data', 없으면 'end'
    """
    result_value = context["task_instance"].xcom_pull(task_ids=task_id, key='result')

    if not result_value:
        return 'end'

    # 중첩된 dict에 값이 있는지 확인
    if isinstance(result_value, dict):
        if expected_keys:
            # 특정 키 확인
            for bank, racks in result_value.items():
                for rack_id, data in racks.items():
                    if all(key in data for key in expected_keys):
                        return 'push_data'
        else:
            # 값이 있는지만 확인
            if any(result_value.values()):
                return 'push_data'

    return 'end'


def check_dataset_for_trigger(**context) -> list:
    """하위 DAG 트리거를 위한 데이터셋이 존재하는지 확인합니다.

    Args:
        **context: Airflow context

    Returns:
        트리거할 task ID 목록 또는 'end'
    """
    result_value = context["task_instance"].xcom_pull(task_ids='query_hi_dataset', key='filename')
    if result_value:
        return ['trigger_TIExVD', 'trigger_PE', 'trigger_MVF', 'trigger_DCIR', 'trigger_VIExTD']
    else:
        return 'end'


def manage_dataset_files(max_files: int = 7, **context) -> None:
    """오래된 파일을 제거하여 데이터셋 파일을 관리합니다.

    Args:
        max_files: 보관할 최대 파일 개수
        **context: Airflow context
    """
    data_save_path = context["task_instance"].xcom_pull(task_ids="initialize_globals", key="DATA_SAVE_PATH")
    f_list = os.listdir(data_save_path)

    if len(f_list) > max_files:
        # 수정 시간 기준으로 파일 정렬
        f_list.sort(key=lambda x: os.path.getmtime(os.path.join(data_save_path, x)))

        # 제거할 파일 선택 (최근 max_files개만 유지)
        files_to_remove = f_list[:-max_files]

        for file in files_to_remove:
            file_path = os.path.join(data_save_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed old file: {file}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"Removed old directory: {file}")
            except Exception as e:
                print(f"Error while deleting {file_path}: {e}")
