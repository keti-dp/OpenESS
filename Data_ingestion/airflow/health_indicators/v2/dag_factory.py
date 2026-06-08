"""건강 지표 DAG를 동적으로 생성하기 위한 DAG factory

이 모듈은 Factory 패턴을 구현하여 다양한 사이트와 건강 지표에 대한
Airflow DAG를 생성하며, 18개 이상의 DAG 파일에서 코드 중복을 제거합니다.
"""

import sys
from pathlib import Path

# DAG 파일의 디렉토리를 Python 경로에 추가
DAG_DIR = Path(__file__).parent.resolve()
if str(DAG_DIR) not in sys.path:
    sys.path.insert(0, str(DAG_DIR))

from functools import partial
from datetime import datetime
import pendulum

from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.dummy import DummyOperator

from config.site_config import get_site_config, DEFAULT_DAG_ARGS, HEALTH_INDICATORS
from operators.common_operators import (
    initialize_global_variables,
    calculate_time_range,
    execute_select_query,
    check_data_for_push,
    check_dataset_for_trigger,
    manage_dataset_files
)
from calculators.dcir_calculator import DCIRCalculator
from calculators.mvf_calculator import MVFCalculator
from calculators.pe_calculator import PECalculator
from calculators.tiexvd_calculator import TIExVDCalculator
from calculators.viextd_calculator import VIExTDCalculator
from utils.database import push_health_indicator_to_database


def create_dataset_dag(site_name: str) -> DAG:
    """사이트의 데이터셋 가져오기 및 저장을 위한 DAG를 생성합니다.

    Args:
        site_name: 사이트 이름 (baekma, gold, panly, seongdeok, seokhwan)

    Returns:
        설정된 Airflow DAG
    """
    site_config = get_site_config(site_name)
    dag_id = f"{site_name}_HI_get_dataset"

    with DAG(
        dag_id=dag_id,
        default_args=DEFAULT_DAG_ARGS,
        start_date=site_config.start_date,
        schedule_interval=None,  # 마스터 DAG에 의해 트리거됨
        tags=[site_name, 'indicator', 'rack', 'dataset'],
        catchup=True
    ) as dag:

        start = DummyOperator(task_id="start")

        initialize_globals = PythonOperator(
            task_id="initialize_globals",
            python_callable=partial(initialize_global_variables, site_config)
        )

        calc_time = PythonOperator(
            task_id='calc_time_range',
            python_callable=calculate_time_range
        )

        query_hi_dataset = PythonOperator(
            task_id='query_hi_dataset',
            python_callable=partial(execute_select_query, site_config)
        )

        branch_operator = BranchPythonOperator(
            task_id='branch_check_data',
            python_callable=check_dataset_for_trigger,
            provide_context=True,
        )

        # 각 건강 지표에 대한 트리거 operator 생성
        triggers = []
        for hi in HEALTH_INDICATORS:
            trigger = TriggerDagRunOperator(
                task_id=f"trigger_{hi}",
                trigger_dag_id=f"{site_name}_HI_calc_{hi}",
                execution_date="{{ execution_date }}",
                conf='{"filename": "{{ task_instance.xcom_pull(task_ids=\'query_hi_dataset\', key=\'filename\') }}"}'
            )
            triggers.append(trigger)

        delete_file = PythonOperator(
            task_id='delete_file',
            python_callable=manage_dataset_files
        )

        end = DummyOperator(task_id="end")
        complete_HI = DummyOperator(task_id="complete_health_indicators")

        # Task 의존성 정의
        start >> initialize_globals >> calc_time >> query_hi_dataset >> branch_operator
        branch_operator >> triggers >> delete_file >> complete_HI
        branch_operator >> end

    return dag


def create_health_indicator_dag(site_name: str, indicator: str) -> DAG:
    """특정 사이트에 대한 특정 건강 지표 계산을 위한 DAG를 생성합니다.

    Args:
        site_name: 사이트 이름 (baekma, gold, panly)
        indicator: 건강 지표 이름 (DCIR, MVF, PE, TIExVD, VIExTD)

    Returns:
        설정된 Airflow DAG
    """
    site_config = get_site_config(site_name)
    dag_id = f"{site_name}_HI_calc_{indicator}"

    # 지표 이름을 계산기 클래스에 매핑
    calculator_map = {
        'DCIR': DCIRCalculator,
        'MVF': MVFCalculator,
        'PE': PECalculator,
        'TIExVD': TIExVDCalculator,
        'VIExTD': VIExTDCalculator
    }

    if indicator not in calculator_map:
        raise ValueError(f"알 수 없는 지표: {indicator}")

    calculator_class = calculator_map[indicator]
    calculator = calculator_class(site_config)

    # Task 이름 지정
    calc_task_id = f'calc_{indicator.lower()}'

    with DAG(
        dag_id=dag_id,
        default_args=DEFAULT_DAG_ARGS,
        start_date=site_config.start_date,
        schedule_interval=None,  # 데이터셋 DAG에 의해 트리거됨
        tags=[site_name, 'indicator', 'rack'],
        catchup=True
    ) as dag:

        start = DummyOperator(task_id="start")

        initialize_globals = PythonOperator(
            task_id="initialize_globals",
            python_callable=partial(initialize_global_variables, site_config)
        )

        calc_time = PythonOperator(
            task_id='calc_time_range',
            python_callable=calculate_time_range
        )

        calc_indicator = PythonOperator(
            task_id=calc_task_id,
            python_callable=calculator.execute_calculation
        )

        branch_operator = BranchPythonOperator(
            task_id='branch_check_data',
            python_callable=partial(check_data_for_push, calc_task_id),
            provide_context=True,
        )

        push_data = PythonOperator(
            task_id='push_data',
            python_callable=partial(push_health_indicator_to_database, calculator, calc_task_id)
        )

        complete_data_push = DummyOperator(task_id="complete_data_push")
        end = DummyOperator(task_id="end")

        # Task 의존성 정의
        start >> initialize_globals >> calc_time >> calc_indicator >> branch_operator
        branch_operator >> push_data >> complete_data_push
        branch_operator >> end

    return dag


def create_master_dag() -> DAG:
    """모든 사이트의 건강 지표 처리를 조율하는 마스터 DAG를 생성합니다.

    이 DAG는 스케줄에 따라 실행되며, 각 사이트의 get_dataset DAG를 트리거합니다.

    Returns:
        설정된 Airflow 마스터 DAG
    """
    dag_id = "Health_Indicator_master"
    sites = ['baekma', 'gold', 'panly', 'seongdeok', 'seokhwan']

    # 가장 빠른 스케줄 사용 (panly: '00 05 * * *')
    # 모든 사이트의 start_date 중 가장 이른 것 사용
    earliest_start_date = min(get_site_config(site).start_date for site in sites)

    with DAG(
        dag_id=dag_id,
        default_args=DEFAULT_DAG_ARGS,
        start_date=earliest_start_date,
        schedule_interval='00 05 * * *',  # 매일 새벽 5시
        tags=['master', 'indicator', 'health_indicator'],
        catchup=True,
        description='모든 사이트의 건강 지표 처리를 조율하는 마스터 DAG'
    ) as dag:

        start = DummyOperator(task_id="start")

        # 각 사이트의 get_dataset DAG를 트리거하는 operator 생성
        site_triggers = []
        for site in sites:
            trigger = TriggerDagRunOperator(
                task_id=f"trigger_{site}_dataset",
                trigger_dag_id=f"{site}_HI_get_dataset",
                execution_date="{{ execution_date }}",
                wait_for_completion=False,  # 병렬 실행을 위해 대기하지 않음
            )
            site_triggers.append(trigger)

        end = DummyOperator(task_id="end")

        # Task 의존성 정의: 모든 사이트를 병렬로 트리거
        start >> site_triggers >> end

    return dag


def generate_all_dags() -> dict:
    """모든 사이트와 건강 지표에 대한 모든 DAG를 생성합니다.

    Returns:
        DAG ID를 DAG 객체에 매핑하는 딕셔너리
    """
    dags = {}
    sites = ['baekma', 'gold', 'panly', 'seongdeok', 'seokhwan']

    # 마스터 DAG 생성 (최우선)
    master_dag = create_master_dag()
    dags[master_dag.dag_id] = master_dag

    # 데이터셋 DAG 생성
    for site in sites:
        dag = create_dataset_dag(site)
        dags[dag.dag_id] = dag

    # 건강 지표 DAG 생성
    for site in sites:
        for indicator in HEALTH_INDICATORS:
            dag = create_health_indicator_dag(site, indicator)
            dags[dag.dag_id] = dag

    return dags
