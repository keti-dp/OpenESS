"""
DAG Factory 모듈

사이트별 설정을 기반으로 Airflow DAG을 동적으로 생성하는 팩토리 함수를 제공합니다.
"""

from datetime import datetime
import pendulum
import sys
import os

# sys.path에 현재 디렉토리 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from config import SiteConfig, get_site_config
from core import TimeCalculator, FileManager
from processors import (
    DatasetProcessor,
    CountProcessor,
    DiffProcessor,
    MovingAvgProcessor,
    InfoProcessor,
)


def create_get_dataset_dag(site_name: str) -> DAG:
    """
    데이터셋 가져오기 DAG 생성

    Args:
        site_name: 사이트 이름 (예: 'baekma', 'gold', 'panly')

    Returns:
        생성된 DAG 객체
    """
    config = get_site_config(site_name)
    kst = pendulum.timezone("Asia/Seoul")

    def initialize_globals(**context):
        """전역 변수 초기화"""
        output_path = config.get_path('original')
        print(f"Initialized OUTPUT_PATH: {output_path}")
        context['task_instance'].xcom_push(key='OUTPUT_PATH', value=output_path)

    def calc_time(**context):
        """시간 계산"""
        time_calc = TimeCalculator()
        query_time = time_calc.calc_day_range(context["execution_date"])
        print(f"Execution_date: {context['execution_date']}")
        print(f"Query time: {query_time}")
        context["task_instance"].xcom_push(key="query_time", value=query_time)

    def get_data(**context):
        """데이터 가져오기 및 저장"""
        query_time = context["task_instance"].xcom_pull(task_ids="calc_time", key="query_time")

        processor = DatasetProcessor(config)
        filename = processor.fetch_and_save_dataset(
            query_time["begin_time"],
            query_time["end_time"]
        )

        if filename:
            context["task_instance"].xcom_push(key="filename", value=filename)

    def check_data(**context):
        """데이터 존재 여부 확인"""
        filename = context["task_instance"].xcom_pull(task_ids='get_data', key='filename')
        print(f"Filename: {filename}")
        if filename:
            return ['trigger_socp_count', 'trigger_socp_info']
        else:
            return 'end'

    def delete_files(**context):
        """오래된 파일 삭제"""
        processor = DatasetProcessor(config)
        processor.cleanup_old_files(max_files=30)

    # DAG 정의
    dag = DAG(
        dag_id=f'{site_name}_SoCP_GetDataset',
        default_args={'owner': config.dag.get('owner', 'airflow')},
        start_date=datetime.strptime(config.dag.get('start_date', '2025-11-10'), '%Y-%m-%d').replace(tzinfo=kst),
        schedule_interval=None,  # master DAG에 의해서만 트리거됨
        tags=config.dag.get('tags', [site_name, 'socp', 'dataset']),
        catchup=config.dag.get('catchup', True),
    )

    with dag:
        start = DummyOperator(task_id="start")

        initialize_globals_task = PythonOperator(
            task_id="initialize_globals",
            python_callable=initialize_globals
        )

        calc_time_task = PythonOperator(
            task_id='calc_time',
            python_callable=calc_time
        )

        get_data_task = PythonOperator(
            task_id='get_data',
            python_callable=get_data,
        )

        branch_check = BranchPythonOperator(
            task_id='branch_check_data',
            python_callable=check_data,
            provide_context=True,
        )

        trigger_count = TriggerDagRunOperator(
            task_id="trigger_socp_count",
            trigger_dag_id=f"{site_name}_SoCP_count",
            execution_date="{{ execution_date }}",
            conf='{"filename": "{{ task_instance.xcom_pull(task_ids=\'get_data\', key=\'filename\') }}"}'
        )

        trigger_info = TriggerDagRunOperator(
            task_id="trigger_socp_info",
            trigger_dag_id=f"{site_name}_SoCP_info",
            execution_date="{{ execution_date }}",
            conf='{"filename": "{{ task_instance.xcom_pull(task_ids=\'get_data\', key=\'filename\') }}"}'
        )

        delete_file_task = PythonOperator(
            task_id='delete_file',
            python_callable=delete_files
        )

        end = DummyOperator(task_id="end")
        complete_socp = DummyOperator(task_id="complete_socp")

        start >> initialize_globals_task >> calc_time_task >> get_data_task >> branch_check
        branch_check >> [trigger_count, trigger_info] >> delete_file_task >> complete_socp
        branch_check >> end

    return dag


def create_count_dag(site_name: str) -> DAG:
    """
    SoCP Count 계산 DAG 생성

    Args:
        site_name: 사이트 이름

    Returns:
        생성된 DAG 객체
    """
    config = get_site_config(site_name)
    kst = pendulum.timezone("Asia/Seoul")

    def initialize_globals(**context):
        """전역 변수 초기화"""
        paths = {
            'ORI_DATA_PATH': config.get_path('original'),
            'WORK_DATA_PATH': config.get_path('count_work'),
            'PREP_DATA_PATH': config.get_path('count_prep'),
        }
        for key, value in paths.items():
            print(f"Initialized {key}: {value}")
            context['task_instance'].xcom_push(key=key, value=value)

    def calc_time(**context):
        """시간 계산"""
        time_calc = TimeCalculator()
        query_time = time_calc.calc_day_range(context["execution_date"])
        print(f"Query time: {query_time}")
        context["task_instance"].xcom_push(key="query_time", value=query_time)

    def prep_dataset(**context):
        """데이터셋 전처리"""
        query_time = context["task_instance"].xcom_pull(task_ids='calc_past_time', key='query_time')
        filename = query_time['begin_time'][:10]

        processor = CountProcessor(config)
        processor.prepare_dataset(filename)

    def calc_socp(**context):
        """SoCP 계산"""
        query_time = context["task_instance"].xcom_pull(task_ids='calc_past_time', key='query_time')
        filename = query_time['begin_time'][:10]
        timestamp = query_time['begin_time']

        processor = CountProcessor(config)
        processor.calculate_count(filename, timestamp)

    def save_database(**context):
        """데이터베이스 저장"""
        query_time = context["task_instance"].xcom_pull(task_ids='calc_past_time', key='query_time')
        filename = query_time['begin_time'][:10]

        processor = CountProcessor(config)
        processor.save_to_database(filename)

    def cleanup_work(**context):
        """작업 파일 정리"""
        processor = CountProcessor(config)
        processor.cleanup_work_files()

    def cleanup_prep(**context):
        """전처리 파일 정리"""
        processor = CountProcessor(config)
        processor.cleanup_prep_files()

    # DAG 정의
    dag = DAG(
        dag_id=f'{site_name}_SoCP_count',
        default_args={'owner': config.dag.get('owner', 'airflow')},
        start_date=datetime.strptime(config.dag.get('start_date', '2025-11-10'), '%Y-%m-%d').replace(tzinfo=kst),
        schedule_interval=None,
        tags=config.dag.get('tags', [site_name, 'socp']),
        catchup=config.dag.get('catchup', True),
    )

    with dag:
        start = DummyOperator(task_id="start")

        initialize_globals_task = PythonOperator(
            task_id="initialize_globals",
            python_callable=initialize_globals
        )

        calc_past_time = PythonOperator(
            task_id='calc_past_time',
            python_callable=calc_time
        )

        prep_dataset_task = PythonOperator(
            task_id='prep_dataset',
            python_callable=prep_dataset
        )

        calc_socp_task = PythonOperator(
            task_id='calc_socp',
            python_callable=calc_socp
        )

        save_database_task = PythonOperator(
            task_id='save_database',
            python_callable=save_database
        )

        delete_work_file = PythonOperator(
            task_id='delete_work_file',
            python_callable=cleanup_work
        )

        delete_prep_file = PythonOperator(
            task_id='delete_prep_file',
            python_callable=cleanup_prep
        )

        connect = DummyOperator(task_id="connect")

        trigger_diff = TriggerDagRunOperator(
            task_id="trigger_differencing",
            trigger_dag_id=f"{site_name}_SoCP_1st_differencing",
            execution_date="{{ execution_date }}"
        )

        trigger_ma = TriggerDagRunOperator(
            task_id="trigger_moving_average",
            trigger_dag_id=f"{site_name}_SoCP_moving_average",
            execution_date="{{ execution_date }}"
        )

        end = DummyOperator(task_id="end")

        start >> initialize_globals_task >> calc_past_time >> prep_dataset_task >> calc_socp_task
        calc_socp_task >> save_database_task >> [delete_work_file, delete_prep_file] >> connect
        connect >> [trigger_diff, trigger_ma] >> end

    return dag


def create_diff_dag(site_name: str) -> DAG:
    """
    1st Differencing DAG 생성

    Args:
        site_name: 사이트 이름

    Returns:
        생성된 DAG 객체
    """
    config = get_site_config(site_name)
    kst = pendulum.timezone("Asia/Seoul")

    def initialize_globals(**context):
        """전역 변수 초기화"""
        save_path = config.get_path('count_diff')
        print(f"Initialized SAVE_DATA_PATH: {save_path}")
        context['task_instance'].xcom_push(key='SAVE_DATA_PATH', value=save_path)

    def calc_time(**context):
        """시간 계산"""
        time_calc = TimeCalculator()
        query_time = time_calc.calc_past_days_for_diff(context["execution_date"])
        print(f"Query time: {query_time}")
        context["task_instance"].xcom_push(key="query_time", value=query_time)

    def get_dataset(**context):
        """데이터셋 가져오기"""
        query_time = context["task_instance"].xcom_pull(task_ids='calc_past_time', key='query_time')
        filename = TimeCalculator.format_filename(query_time['end'])

        processor = DiffProcessor(config)
        processor.fetch_source_data(
            query_time['start'],
            query_time['end'],
            filename
        )
        context["task_instance"].xcom_push(key="f_name", value=filename)

    def calc_diff(**context):
        """차분 계산"""
        filename = context["task_instance"].xcom_pull(task_ids='getDataset', key='f_name')

        processor = DiffProcessor(config)
        processor.calculate_differencing(filename)

    def push_data(**context):
        """데이터베이스 저장"""
        filename = context["task_instance"].xcom_pull(task_ids='getDataset', key='f_name')

        processor = DiffProcessor(config)
        processor.save_to_database(filename)

    def cleanup(**context):
        """파일 정리"""
        processor = DiffProcessor(config)
        processor.cleanup_files()

    # DAG 정의
    dag = DAG(
        dag_id=f'{site_name}_SoCP_1st_differencing',
        default_args={'owner': config.dag.get('owner', 'airflow')},
        start_date=datetime.strptime(config.dag.get('start_date', '2025-11-10'), '%Y-%m-%d').replace(tzinfo=kst),
        schedule_interval=None,
        tags=config.dag.get('tags', [site_name, 'socp']),
        catchup=config.dag.get('catchup', True),
    )

    with dag:
        start = DummyOperator(task_id="start")

        initialize_globals_task = PythonOperator(
            task_id="initialize_globals",
            python_callable=initialize_globals
        )

        calc_past_time = PythonOperator(
            task_id='calc_past_time',
            python_callable=calc_time
        )

        get_dataset_task = PythonOperator(
            task_id='getDataset',
            python_callable=get_dataset,
        )

        calc_1st_diff = PythonOperator(
            task_id='calc_1st_diff',
            python_callable=calc_diff,
        )

        push_data_task = PythonOperator(
            task_id='push_data',
            python_callable=push_data,
        )

        delete_file = PythonOperator(
            task_id='delete_file',
            python_callable=cleanup
        )

        end = DummyOperator(task_id="end")

        start >> initialize_globals_task >> calc_past_time >> get_dataset_task
        get_dataset_task >> calc_1st_diff >> push_data_task >> delete_file >> end

    return dag


def create_moving_avg_dag(site_name: str) -> DAG:
    """
    Moving Average DAG 생성

    Args:
        site_name: 사이트 이름

    Returns:
        생성된 DAG 객체
    """
    config = get_site_config(site_name)
    kst = pendulum.timezone("Asia/Seoul")

    def initialize_globals(**context):
        """전역 변수 초기화"""
        save_path = config.get_path('moving_avg')
        print(f"Initialized SAVE_DATA_PATH: {save_path}")
        context['task_instance'].xcom_push(key='SAVE_DATA_PATH', value=save_path)

    def calc_time(**context):
        """시간 계산"""
        time_calc = TimeCalculator()
        execution_date = context["execution_date"]
        seoul_time = time_calc.convert_to_seoul(execution_date)

        # 5일, 10일, 15일, 30일 전 계산
        past_5days = (seoul_time - pendulum.duration(days=4)).start_of('day').to_datetime_string()
        past_10days = (seoul_time - pendulum.duration(days=9)).start_of('day').to_datetime_string()
        past_15days = (seoul_time - pendulum.duration(days=14)).start_of('day').to_datetime_string()
        past_30days = (seoul_time - pendulum.duration(days=29)).start_of('day').to_datetime_string()
        end = seoul_time.start_of('day').to_datetime_string()

        query_time = {
            "past_5days": past_5days,
            "past_10days": past_10days,
            "past_15days": past_15days,
            "past_30days": past_30days,
            "end": end
        }
        print(query_time)
        context["task_instance"].xcom_push(key="query_time", value=query_time)

    def get_dataset(period_name: str, ma_value: int):
        """데이터셋 가져오기 함수 생성"""
        def _get_dataset(**context):
            query_time = context["task_instance"].xcom_pull(task_ids='calc_past_time', key='query_time')
            filename = TimeCalculator.format_filename(query_time['end'])

            processor = MovingAvgProcessor(config)
            processor.fetch_source_data(
                query_time[period_name],
                query_time['end'],
                filename,
                ma_value
            )
            context["task_instance"].xcom_push(key="f_name", value=filename)
        return _get_dataset

    def calc_ma(ma_value: int):
        """Moving Average 계산 함수 생성"""
        def _calc_ma(**context):
            filename = context["task_instance"].xcom_pull(task_ids=f'getDataset_{ma_value}', key='f_name')

            processor = MovingAvgProcessor(config)
            processor.calculate_moving_average(filename, ma_value)
        return _calc_ma

    def push_data(**context):
        """데이터베이스 저장"""
        filename = context["task_instance"].xcom_pull(task_ids='getDataset_5', key='f_name')

        processor = MovingAvgProcessor(config)
        processor.save_to_database(filename)

    def cleanup(**context):
        """파일 정리"""
        processor = MovingAvgProcessor(config)
        processor.cleanup_files()

    # DAG 정의
    dag = DAG(
        dag_id=f'{site_name}_SoCP_moving_average',
        default_args={'owner': config.dag.get('owner', 'airflow')},
        start_date=datetime.strptime(config.dag.get('start_date', '2025-11-10'), '%Y-%m-%d').replace(tzinfo=kst),
        schedule_interval=None,
        tags=config.dag.get('tags', [site_name, 'socp']),
        catchup=config.dag.get('catchup', True),
    )

    with dag:
        start = DummyOperator(task_id="start")

        initialize_globals_task = PythonOperator(
            task_id="initialize_globals",
            python_callable=initialize_globals
        )

        calc_past_time = PythonOperator(
            task_id='calc_past_time',
            python_callable=calc_time
        )

        # 각 MA 윈도우별 태스크 생성
        ma_configs = [
            ('past_5days', 5),
            ('past_10days', 10),
            ('past_15days', 15),
            ('past_30days', 30)
        ]

        get_tasks = []
        calc_tasks = []

        for period_name, ma_value in ma_configs:
            get_task = PythonOperator(
                task_id=f'getDataset_{ma_value}',
                python_callable=get_dataset(period_name, ma_value),
            )

            calc_task = PythonOperator(
                task_id=f'calc_{ma_value}MA',
                python_callable=calc_ma(ma_value),
            )

            get_tasks.append(get_task)
            calc_tasks.append(calc_task)

            calc_past_time >> get_task >> calc_task

        push_data_task = PythonOperator(
            task_id='push_data',
            python_callable=push_data,
        )

        delete_file = PythonOperator(
            task_id='delete_file',
            python_callable=cleanup
        )

        end = DummyOperator(task_id="end")

        start >> initialize_globals_task >> calc_past_time
        calc_tasks >> push_data_task >> delete_file >> end

    return dag


def create_info_dag(site_name: str) -> DAG:
    """
    SoCP Info DAG 생성

    Args:
        site_name: 사이트 이름

    Returns:
        생성된 DAG 객체
    """
    config = get_site_config(site_name)
    kst = pendulum.timezone("Asia/Seoul")

    def initialize_globals(**context):
        """전역 변수 초기화"""
        paths = {
            'LOAD_DATA_PATH': config.get_path('original'),
            'PERIOD_DATA_PATH': config.get_path('info_period'),
            'PREP_DATA_PATH': config.get_path('info_prep'),
        }
        for key, value in paths.items():
            print(f"Initialized {key}: {value}")
            context['task_instance'].xcom_push(key=key, value=value)

    def calc_time(**context):
        """시간 계산"""
        time_calc = TimeCalculator()
        query_time = time_calc.calc_period_ranges(context["execution_date"])
        print(f"Query time: {query_time}")
        context["task_instance"].xcom_push(key="query_time", value=query_time)

    def get_dataset(period_name: str):
        """데이터셋 로드 함수 생성"""
        def _get_dataset(**context):
            query_time = context["task_instance"].xcom_pull(task_ids='calc_time', key='query_time')

            processor = InfoProcessor(config)
            processor.load_and_split_dataset(
                query_time[period_name],
                query_time['end_time'],
                period_name
            )
        return _get_dataset

    def calc_socp(**context):
        """SoCP 계산"""
        query_time = context["task_instance"].xcom_pull(task_ids='calc_time', key='query_time')
        filename = query_time['end_time'][:10]
        timestamp = query_time['past_1days']

        processor = InfoProcessor(config)

        # 각 기간별로 계산
        results = {}
        period_configs = [
            ('past_1days', 1),
            ('past_7days', 7),
            ('past_30days', 30)
        ]

        for period_name, period_value in period_configs:
            df = processor.calculate_info(filename, period_name, period_value, timestamp)
            results[period_name] = df

        # 결과 병합 및 저장
        processor.merge_and_save_results(filename, results)

    def save_database(**context):
        """데이터베이스 저장"""
        query_time = context["task_instance"].xcom_pull(task_ids='calc_time', key='query_time')
        filename = query_time['end_time'][:10]

        processor = InfoProcessor(config)
        processor.save_to_database(filename)

    def cleanup_period(**context):
        """기간별 파일 정리"""
        processor = InfoProcessor(config)
        processor.cleanup_period_files()

    def cleanup_prep(**context):
        """전처리 파일 정리"""
        processor = InfoProcessor(config)
        processor.cleanup_prep_files()

    # DAG 정의
    dag = DAG(
        dag_id=f'{site_name}_SoCP_info',
        default_args={'owner': config.dag.get('owner', 'airflow')},
        start_date=datetime.strptime(config.dag.get('start_date', '2025-11-10'), '%Y-%m-%d').replace(tzinfo=kst),
        schedule_interval=None,
        tags=config.dag.get('tags', [site_name, 'socp']),
        catchup=config.dag.get('catchup', True),
    )

    with dag:
        start = DummyOperator(task_id="start")

        initialize_globals_task = PythonOperator(
            task_id="initialize_globals",
            python_callable=initialize_globals
        )

        calc_time_task = PythonOperator(
            task_id='calc_time',
            python_callable=calc_time
        )

        # 각 기간별 데이터 로드 태스크
        get_1day = PythonOperator(
            task_id='get_dataset_1day',
            python_callable=get_dataset('past_1days')
        )

        get_7days = PythonOperator(
            task_id='get_dataset_7days',
            python_callable=get_dataset('past_7days')
        )

        get_30days = PythonOperator(
            task_id='get_dataset_30days',
            python_callable=get_dataset('past_30days')
        )

        calc_socp_task = PythonOperator(
            task_id='calc_socp',
            python_callable=calc_socp
        )

        save_database_task = PythonOperator(
            task_id='save_database',
            python_callable=save_database
        )

        delete_period_file = PythonOperator(
            task_id='delete_period_file',
            python_callable=cleanup_period
        )

        delete_prep_file = PythonOperator(
            task_id='delete_prep_file',
            python_callable=cleanup_prep
        )

        end = DummyOperator(task_id="end")

        start >> initialize_globals_task >> calc_time_task
        calc_time_task >> [get_1day, get_7days, get_30days] >> calc_socp_task
        calc_socp_task >> save_database_task >> [delete_period_file, delete_prep_file] >> end

    return dag


def create_master_dag() -> DAG:
    """
    모든 사이트의 SoCP 처리를 조율하는 마스터 DAG 생성

    이 DAG는 스케줄에 따라 실행되며, 각 사이트의 GetDataset DAG를 트리거합니다.

    Returns:
        생성된 마스터 DAG 객체
    """
    from config.site_config import SiteConfigLoader

    dag_id = "SoCP_master"
    kst = pendulum.timezone("Asia/Seoul")

    # 모든 사이트 목록 가져오기
    loader = SiteConfigLoader()
    sites = loader.get_all_sites()

    # 모든 사이트의 설정 가져오기
    site_configs = [loader.get_config(site) for site in sites]

    # 가장 이른 스케줄 찾기 (panly: '00 4 * * *')
    # 모든 사이트에서 동일한 start_date 사용
    earliest_start_date = datetime.strptime(
        site_configs[0].dag.get('start_date', '2025-11-10'),
        '%Y-%m-%d'
    ).replace(tzinfo=kst)

    # 가장 이른 스케줄 사용
    schedule_times = []
    for config in site_configs:
        schedule = config.dag.get('schedule_get_dataset', '10 4 * * *')
        schedule_times.append(schedule)

    # 가장 이른 시간 찾기 (00 4 * * *)
    earliest_schedule = min(schedule_times)

    # DAG 정의
    dag = DAG(
        dag_id=dag_id,
        default_args={'owner': 'jwpark'},
        start_date=earliest_start_date,
        schedule_interval=earliest_schedule,
        tags=['master', 'socp'],
        catchup=True,
        description='모든 사이트의 SoCP 처리를 조율하는 마스터 DAG'
    )

    with dag:
        start = DummyOperator(task_id="start")

        # 각 사이트의 GetDataset DAG를 트리거하는 operator 생성
        site_triggers = []
        for site in sites:
            trigger = TriggerDagRunOperator(
                task_id=f"trigger_{site}_dataset",
                trigger_dag_id=f"{site}_SoCP_GetDataset",
                execution_date="{{ execution_date }}",
                wait_for_completion=False,  # 병렬 실행을 위해 대기하지 않음
            )
            site_triggers.append(trigger)

        end = DummyOperator(task_id="end")

        # Task 의존성 정의: 모든 사이트를 병렬로 트리거
        start >> site_triggers >> end

    return dag


def generate_all_dags() -> dict:
    """
    모든 사이트와 SoCP DAG 유형에 대한 모든 DAG를 생성합니다.

    Returns:
        DAG ID를 DAG 객체에 매핑하는 딕셔너리
    """
    from config.site_config import SiteConfigLoader

    dags = {}
    loader = SiteConfigLoader()
    sites = loader.get_all_sites()

    # 마스터 DAG 생성 (최우선)
    master_dag = create_master_dag()
    dags[master_dag.dag_id] = master_dag

    # 각 사이트별로 모든 DAG 생성
    for site in sites:
        # GetDataset DAG 생성
        get_dataset_dag = create_get_dataset_dag(site)
        dags[get_dataset_dag.dag_id] = get_dataset_dag

        # Count DAG 생성
        count_dag = create_count_dag(site)
        dags[count_dag.dag_id] = count_dag

        # Info DAG 생성
        info_dag = create_info_dag(site)
        dags[info_dag.dag_id] = info_dag

        # 1st Differencing DAG 생성
        diff_dag = create_diff_dag(site)
        dags[diff_dag.dag_id] = diff_dag

        # Moving Average DAG 생성
        ma_dag = create_moving_avg_dag(site)
        dags[ma_dag.dag_id] = ma_dag

    return dags
