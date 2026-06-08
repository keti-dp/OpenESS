"""여러 DAG에서 사용되는 공통 operator 함수들"""

import sys
from pathlib import Path

# DAG 파일의 디렉토리를 Python 경로에 추가
DAG_DIR = Path(__file__).parent.parent.resolve()
if str(DAG_DIR) not in sys.path:
    sys.path.insert(0, str(DAG_DIR))

import os
import shutil
from typing import Dict, Any, List

from config import SiteConfig
from core import TimeCalculator


def initialize_global_variables(site_config: SiteConfig, **context) -> None:
    """전역 변수를 초기화하고 XCom에 push합니다.

    Args:
        site_config: 사이트 설정 객체
        **context: Airflow context
    """
    output_path = site_config.get_path('original')
    print(f"Initialized OUTPUT_PATH: {output_path}")
    context['task_instance'].xcom_push(key='OUTPUT_PATH', value=output_path)


def calculate_time_range(**context) -> None:
    """데이터 쿼리를 위한 시간 범위를 계산하고 XCom에 push합니다.

    Args:
        **context: Airflow context
    """
    time_calc = TimeCalculator()
    query_time = time_calc.calc_day_range(context["execution_date"])

    print(f"Execution_date: {context['execution_date']}")
    print(f"Query time range: {query_time}")

    context["task_instance"].xcom_push(key="query_time", value=query_time)


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
            for key in expected_keys:
                if key in result_value and result_value[key]:
                    return 'push_data'
        else:
            # 값이 있는지만 확인
            if any(result_value.values()):
                return 'push_data'

    return 'end'


def check_dataset_for_trigger(**context) -> List[str]:
    """하위 DAG 트리거를 위한 데이터셋이 존재하는지 확인합니다.

    Args:
        **context: Airflow context

    Returns:
        트리거할 task ID 목록 또는 'end'
    """
    result_value = context["task_instance"].xcom_pull(task_ids='get_data', key='filename')
    if result_value:
        return ['trigger_socp_count', 'trigger_socp_info']
    else:
        return 'end'


def manage_dataset_files(max_files: int = 30, **context) -> None:
    """오래된 파일을 제거하여 데이터셋 파일을 관리합니다.

    Args:
        max_files: 보관할 최대 파일 개수
        **context: Airflow context
    """
    output_path = context["task_instance"].xcom_pull(task_ids="initialize_globals", key="OUTPUT_PATH")

    if not os.path.exists(output_path):
        print(f"Path does not exist: {output_path}")
        return

    f_list = os.listdir(output_path)

    if len(f_list) > max_files:
        # 수정 시간 기준으로 파일 정렬
        f_list.sort(key=lambda x: os.path.getmtime(os.path.join(output_path, x)))

        # 제거할 파일 선택 (최근 max_files개만 유지)
        files_to_remove = f_list[:-max_files]

        for file in files_to_remove:
            file_path = os.path.join(output_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed old file: {file}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"Removed old directory: {file}")
            except Exception as e:
                print(f"Error while deleting {file_path}: {e}")
    else:
        print(f"File count ({len(f_list)}) is within limit ({max_files})")
