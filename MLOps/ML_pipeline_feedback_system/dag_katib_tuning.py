"""
Katib 하이퍼파라미터 튜닝 DAG (v1)
매월 1일에 실행하여 최적 하이퍼파라미터를 탐색하고 베스트 모델을 GCS에 저장합니다.

실행 흐름:
1. Katib 파이프라인 실행 (하이퍼파라미터 튜닝)
2. 각 Trial에서 모델 학습 및 GCS 저장
3. 베스트 하이퍼파라미터와 베스트 모델 GCS에 저장
4. 결과 확인 및 알림 전송

저장 위치:
- 하이퍼파라미터: gs://keti-airflow-dataset/vt-model/{site_id}/models/{yyyymm}/{yyyymm}_xgboost_{site_id}.json
- 베스트 모델: gs://keti-airflow-dataset/vt-model/{site_id}/models/{yyyymm}/{yyyymm}_xgboost_{site_id}_model.pkl
- 모든 Trial 모델: gs://keti-airflow-dataset/vt-model/{site_id}/models/{yyyymm}/trials/
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import pendulum
from pathlib import Path
import sys
import os
import json
import yaml

# 현재 디렉토리를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

# 사이트 ID 지정 (환경변수에서 읽거나 기본값 사용)
SITE_ID = os.getenv('SITE_ID', 'default_site')

# katib_config.yaml 로드
config_path = Path(__file__).parent / 'config' / 'katib_config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

if SITE_ID not in config['sites']:
    raise ValueError(f"사이트 '{SITE_ID}'를 찾을 수 없습니다. 사용 가능한 사이트: {list(config['sites'].keys())}")

site_config = config['sites'][SITE_ID]
airflow_config = config['defaults']['airflow']
kubeflow_config = config['defaults']['kubeflow']


def run_katib_tuning(**context):
    """Katib 하이퍼파라미터 튜닝 파이프라인 실행"""
    import tempfile
    from kfp import compiler
    from utils.kubeflow_client import KubeflowClient
    from pipeline.katib_pipeline import katib_tuning_pipeline

    ti = context['task_instance']
    execution_date = context['execution_date']

    print(f"[{site_config['name']}] Katib 하이퍼파라미터 튜닝 시작")
    print(f"실행 날짜: {execution_date.strftime('%Y-%m-%d')}")

    # katib_config.yaml의 kubeflow 설정에서 파라미터 구성
    early_stopping_config = kubeflow_config.get('early_stopping', {})

    katib_params = {
        'namespace': kubeflow_config.get('namespace', 'space-openess'),
        'experiment_name_prefix': SITE_ID,
        'timeout': kubeflow_config.get('katib_timeout', 7200),
        'max_trial_count': kubeflow_config.get('max_trial_count', 30),
        'parallel_trial_count': kubeflow_config.get('parallel_trial_count', 3),
        'training_image': kubeflow_config.get('training_image', 'ghcr.io/keti-dp/openess-public:keti.ai_maxvol_xgboost_models-0.6'),
        'parameters_config': json.dumps(kubeflow_config.get('katib_parameters', {})),
        # Early Stopping 설정
        'early_stopping_enabled': early_stopping_config.get('enabled', True),
        'early_stopping_algorithm': early_stopping_config.get('algorithm', 'medianstop'),
        'early_stopping_min_trials': early_stopping_config.get('min_trials_required', 3),
        'early_stopping_start_step': early_stopping_config.get('start_step', 5)
    }

    print("\nKatib 설정 (katib_config.yaml에서 로드):")
    print(f"  - Namespace: {katib_params['namespace']}")
    print(f"  - Experiment Prefix: {katib_params['experiment_name_prefix']}")
    print(f"  - Max trials: {katib_params['max_trial_count']}")
    print(f"  - Parallel trials: {katib_params['parallel_trial_count']}")
    print(f"  - Timeout: {katib_params['timeout']}s ({katib_params['timeout']//3600}h)")
    print(f"  - Early Stopping: {'Enabled' if katib_params['early_stopping_enabled'] else 'Disabled'}")
    if katib_params['early_stopping_enabled']:
        print(f"    - Algorithm: {katib_params['early_stopping_algorithm']}")
        print(f"    - Min trials: {katib_params['early_stopping_min_trials']}")
        print(f"    - Start step: {katib_params['early_stopping_start_step']}")
    print(f"  - 결과 저장: PVC (/mnt/ess-dataset/{SITE_ID}/models/yyyymm/)")

    # Kubeflow 클라이언트 생성
    kf_client = KubeflowClient()

    # 파이프라인 컴파일
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline_path = os.path.join(tmpdir, 'katib_tuning_pipeline.yaml')

        print(f"\n파이프라인 컴파일 중...")
        compiler.Compiler().compile(
            pipeline_func=katib_tuning_pipeline,
            package_path=pipeline_path
        )
        print(f"✓ 파이프라인 컴파일 완료")

        # 파이프라인 실행
        # Use simpler run_name to avoid MySQL collation issues
        import time
        run = kf_client.create_run(
            pipeline_path=pipeline_path,
            experiment_name=f'katib-tuning-{SITE_ID}',
            run_name=f"katib{int(time.time())}",  # Simpler name with timestamp
            params=katib_params
        )

    print(f"✓ 파이프라인 제출 완료")
    print(f"  Run ID: {run.run_id}")

    # 완료 대기
    print(f"\n파이프라인 완료 대기 중... (최대 {katib_params['timeout']//3600}시간)")
    status = kf_client.wait_for_run_completion(run.run_id, timeout=katib_params['timeout'])

    # 결과 확인
    success = status['status'] == 'SUCCEEDED' if status else False

    if not success:
        print(f"\n⚠️ Katib 튜닝 실패: {status['status'] if status else 'UNKNOWN'}")
        print("기본 하이퍼파라미터를 사용하세요.")
    else:
        print(f"\n✓ Katib 튜닝 완료!")
        print(f"  Run ID: {run.run_id}")
        print(f"  Status: {status['status']}")

    # 결과를 XCom에 저장
    ti.xcom_push(key='katib_success', value=success)
    ti.xcom_push(key='run_id', value=run.run_id)
    ti.xcom_push(key='status', value=status['status'] if status else 'UNKNOWN')

    return {
        'success': success,
        'run_id': run.run_id,
        'status': status['status'] if status else 'UNKNOWN'
    }


def save_results(**context):
    """Katib 결과를 로컬에 저장하고 GCS에서 최적 파라미터 확인"""
    from google.cloud import storage

    ti = context['task_instance']
    execution_date = context['execution_date']

    katib_success = ti.xcom_pull(key='katib_success', task_ids='run_katib_tuning')
    run_id = ti.xcom_pull(key='run_id', task_ids='run_katib_tuning')

    print(f"\n{'='*60}")
    print(f"Katib 튜닝 결과")
    print(f"{'='*60}")
    print(f"성공 여부: {'✓ 성공' if katib_success else '✗ 실패'}")
    print(f"Run ID: {run_id}")
    print(f"실행 날짜: {execution_date.strftime('%Y-%m-%d')}")

    # GCS에서 최적 파라미터 확인
    best_params = None
    if katib_success:
        try:
            # GCS 경로: gs://{bucket}/{model_base_path}/{site_id}/models/yyyymm/
            yearmonth = execution_date.strftime('%Y%m')
            filename = f"{yearmonth}_xgboost_{SITE_ID}.json"
            gcs_bucket = os.getenv('GCS_BUCKET', 'your-gcs-bucket')
            model_base_path = os.getenv('GCS_MODEL_BASE_PATH', 'vt-model')
            gcs_path = f"{model_base_path}/{SITE_ID}/models/{yearmonth}/{filename}"

            print(f"\n📥 GCS에서 최적 파라미터 로드 중...")
            print(f"  경로: gs://{gcs_bucket}/{gcs_path}")

            storage_client = storage.Client()
            bucket = storage_client.bucket(gcs_bucket)
            blob = bucket.blob(gcs_path)

            if blob.exists():
                best_params = json.loads(blob.download_as_string())

                print(f"  ✓ 최적 파라미터 로드 완료")
                print(f"\n최적 하이퍼파라미터:")
                print(f"  {json.dumps(best_params['parameters'], indent=2)}")
                print(f"\n성능 메트릭:")
                print(f"  {json.dumps(best_params['metrics'], indent=2)}")
            else:
                print(f"  ⚠️ 최적 파라미터 파일을 찾을 수 없습니다.")
                print(f"  파일: gs://{gcs_bucket}/{gcs_path}")
        except Exception as e:
            print(f"  ⚠️ 최적 파라미터 로드 실패: {e}")

    if katib_success and best_params:
        print(f"\n✓ Katib 튜닝 완료!")
        print(f"  - 베스트 하이퍼파라미터: GCS에 저장됨")
        print(f"  - 베스트 모델: GCS에 저장됨")
        kubeflow_ui = os.getenv('KUBEFLOW_UI', 'https://your-kubeflow-url')
        print(f"  - Kubeflow UI: {kubeflow_ui}")
    elif not katib_success:
        print(f"\n권장 사항:")
        print(f"  - Katib 실험 로그 확인")
        print(f"  - 다시 실행 시도")

    print(f"{'='*60}\n")

    # 결과 파일로 저장
    result = {
        'site_id': SITE_ID,
        'site_name': site_config['name'],
        'execution_date': execution_date.strftime('%Y-%m-%d'),
        'run_id': run_id,
        'success': katib_success,
        'best_params': best_params['parameters'] if best_params else None,
        'metrics': best_params['metrics'] if best_params else None,
        'timestamp': datetime.now().isoformat()
    }

    output_dir = Path('/tmp/katib_results')
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"katib_{SITE_ID}_{execution_date.strftime('%Y%m')}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✓ 결과 저장: {output_file}")

    return result


def send_notification(**context):
    """완료 알림 전송"""
    ti = context['task_instance']
    execution_date = context['execution_date']

    result = ti.xcom_pull(key='return_value', task_ids='save_results')

    message = f"""
{'='*60}
Katib 하이퍼파라미터 튜닝 완료
{'='*60}

사이트: {result['site_name']} ({result['site_id']})
실행 날짜: {result['execution_date']}
상태: {'✓ 성공' if result['success'] else '✗ 실패'}
Run ID: {result['run_id']}

Kubeflow UI: {os.getenv('KUBEFLOW_UI', 'https://your-kubeflow-url')}

{'='*60}
"""

    print(message)

    # 여기에 이메일, Slack 등의 알림을 추가할 수 있습니다

    return message


# DAG 기본 인자
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': airflow_config.get('retries', 1),
    'retry_delay': timedelta(minutes=airflow_config.get('retry_delay_minutes', 30)),
    'execution_timeout': timedelta(hours=3),  # Katib만 실행하므로 3시간으로 단축
}

# DAG 생성
dag = DAG(
    f'katib_tuning_{SITE_ID}',
    default_args=default_args,
    description=f'[{site_config["name"]}] Katib 하이퍼파라미터 튜닝',
    schedule=airflow_config.get('monthly_training_schedule', '0 3 1 * *'),  # 매월 1일 오전 3시
    start_date=pendulum.datetime(2025, 1, 1, tz='Asia/Seoul'),
    catchup=False,
    tags=['battery', 'katib', 'hyperparameter-tuning', SITE_ID],
    max_active_runs=airflow_config.get('max_active_runs', 1),
)

# Task 정의
katib_task = PythonOperator(
    task_id='run_katib_tuning',
    python_callable=run_katib_tuning,
    provide_context=True,
    dag=dag,
)

save_task = PythonOperator(
    task_id='save_results',
    python_callable=save_results,
    provide_context=True,
    dag=dag,
)

notification_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    provide_context=True,
    dag=dag,
)

# 모델 검증 및 배포 DAG 트리거
trigger_deployment = TriggerDagRunOperator(
    task_id='trigger_model_deployment',
    trigger_dag_id=f'model_validation_deployment_{SITE_ID}',
    wait_for_completion=False,  # 비동기로 실행 (Deploy DAG가 독립적으로 실행)
    dag=dag,
)

# Task 의존성
katib_task >> save_task >> notification_task >> trigger_deployment
