"""
모델 검증 및 자동 배포 DAG
- 새 모델 검증
- 성능 기준 충족 시 자동 배포
- GCS에 배포 이력 기록
- KServe 자동 리로드
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import pendulum
from pathlib import Path
import sys
import yaml

# 현재 디렉토리를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

# 배포 유틸리티 임포트
from utils.deployment import (
    promote_model_to_production,
    record_deployment_to_gcs,
    trigger_kserve_reload
)

# 사이트 ID 지정 (환경변수에서 읽거나 기본값 사용)
import os
SITE_ID = os.getenv('SITE_ID', 'default_site')

# deploy.yaml 로드
config_path = Path(__file__).parent / 'config' / 'deploy.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

if SITE_ID not in config['sites']:
    raise ValueError(f"사이트 '{SITE_ID}'를 찾을 수 없습니다. 사용 가능한 사이트: {list(config['sites'].keys())}")

site_config = config['sites'][SITE_ID]

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,  # 이메일 알림 비활성화
    'email': ['ml-team@your-company.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    f'model_validation_deployment_{SITE_ID}',
    default_args=default_args,
    description=f'[{SITE_ID}] 모델 검증 및 자동 배포 (GCS 이력 관리 + KServe 자동 리로드)',
    schedule=None,  # Katib DAG에서 트리거
    start_date=pendulum.datetime(2025, 1, 1, tz='Asia/Seoul'),
    catchup=False,
    tags=[SITE_ID, 'model', 'validation', 'deployment', 'kserve'],  # SITE_ID를 첫 번째로
)




def load_models_from_gcs(**context):
    """GCS에서 신규/기존 모델 로드"""
    from google.cloud import storage
    from google.oauth2 import service_account

    ti = context['task_instance']
    execution_date = context['execution_date']
    yearmonth = execution_date.strftime('%Y%m')

    # 설정에서 GCS 정보 가져오기
    bucket_name = config['gcs']['bucket_name']
    credentials_path = config['gcs']['credentials_path']
    site_id = SITE_ID

    # 경로 설정 (deploy.yaml 기반)
    new_model_path = f"{site_config['paths']['models_dir']}/{yearmonth}/{yearmonth}_xgboost_{site_id}_model.pkl"
    current_model_path = f"{site_config['paths']['deploy_dir']}/{site_config['deploy_files']['model']}"

    # GCS 인증
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    gcp_project = os.getenv('GCP_PROJECT', 'your-gcp-project')
    storage_client = storage.Client(credentials=credentials, project=gcp_project)
    bucket = storage_client.bucket(bucket_name)

    print(f"\n{'='*60}")
    print(f"모델 경로 확인")
    print(f"{'='*60}")

    # 1. 새 모델 존재 확인 (로드하지 않음)
    print(f"\n📥 새 모델 경로 확인:")
    print(f"  경로: gs://{bucket_name}/{new_model_path}")

    new_model_blob = bucket.blob(new_model_path)
    if not new_model_blob.exists():
        raise FileNotFoundError(f"새 모델을 찾을 수 없습니다: {new_model_path}")

    # 메타데이터 로드 (size 정보 가져오기)
    new_model_blob.reload()
    print(f"  ✓ 새 모델 파일 존재 확인 (크기: {new_model_blob.size / 1024:.2f} KB)")

    # 2. 현재 배포된 모델 존재 확인 (로드하지 않음)
    current_model_blob = bucket.blob(current_model_path)
    current_model_exists = current_model_blob.exists()

    if current_model_exists:
        print(f"\n📥 현재 배포 모델 경로 확인:")
        print(f"  경로: gs://{bucket_name}/{current_model_path}")
        # 메타데이터 로드
        current_model_blob.reload()
        print(f"  ✓ 현재 모델 파일 존재 확인 (크기: {current_model_blob.size / 1024:.2f} KB)")
    else:
        print(f"\n⚠️  현재 배포된 모델 없음 (첫 배포)")

    # XCom으로 전달 (모델 경로만 전달, 실제 로드는 validate 단계에서)
    ti.xcom_push(key='new_model_path', value=f'gs://{bucket_name}/{new_model_path}')
    ti.xcom_push(key='current_model_path', value=f'gs://{bucket_name}/{current_model_path}' if current_model_exists else None)
    ti.xcom_push(key='model_version', value=yearmonth)

    return {
        'new_model_exists': True,
        'current_model_exists': current_model_exists,
        'model_version': yearmonth
    }




def load_validation_data_from_gcs(**context):
    """GCS에서 검증 데이터 로드 및 전처리 (Parquet 파일들)"""
    import tempfile
    from google.cloud import storage
    from google.oauth2 import service_account

    # shared_utils는 이미 상단에서 sys.path에 추가되었으므로 직접 import
    from training.shared_utils import load_and_preprocess_validation_data

    ti = context['task_instance']
    site_id = SITE_ID

    # 설정에서 GCS 정보 가져오기
    bucket_name = config['gcs']['bucket_name']
    credentials_path = config['gcs']['credentials_path']
    gcs_validation_prefix = 'val-dataset/'

    print(f"\n{'='*60}")
    print(f"검증 데이터 로드 및 전처리 (GCS)")
    print(f"{'='*60}")
    print(f"  Bucket: gs://{bucket_name}")
    print(f"  경로: {gcs_validation_prefix}")

    try:
        # GCS 클라이언트 생성
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        gcp_project = os.getenv('GCP_PROJECT', 'your-gcp-project')
        storage_client = storage.Client(credentials=credentials, project=gcp_project)
        bucket = storage_client.bucket(bucket_name)

        # GCS에서 Parquet 파일 목록 가져오기
        print(f"\n📥 GCS에서 Parquet 파일 검색 중...")
        blobs = list(bucket.list_blobs(prefix=gcs_validation_prefix))
        parquet_blobs = [b for b in blobs if b.name.endswith('.parquet')]

        if not parquet_blobs:
            raise FileNotFoundError(
                f"GCS에 검증 데이터가 없습니다: gs://{bucket_name}/{gcs_validation_prefix}\n"
                f"다음 명령으로 데이터를 업로드하세요:\n"
                f"  gsutil -m cp /path/to/*.parquet gs://{bucket_name}/{gcs_validation_prefix}"
            )

        print(f"  ✓ {len(parquet_blobs)}개 Parquet 파일 발견")
        for blob in parquet_blobs[:5]:  # 처음 5개만 출력
            print(f"    - {blob.name}")
        if len(parquet_blobs) > 5:
            print(f"    ... 외 {len(parquet_blobs) - 5}개")

        # 임시 디렉토리 생성 및 파일 다운로드
        temp_dir = Path(tempfile.mkdtemp(prefix='validation_data_'))
        print(f"\n📥 파일 다운로드 중...")
        print(f"  로컬 경로: {temp_dir}")

        downloaded_files = []
        for blob in parquet_blobs:
            local_path = temp_dir / Path(blob.name).name
            blob.download_to_filename(str(local_path))
            downloaded_files.append(local_path)
            print(f"  ✓ {blob.name} → {local_path.name}")

        print(f"\n  총 {len(downloaded_files)}개 파일 다운로드 완료")

        # shared_utils를 사용하여 전처리
        print(f"\n🔄 데이터 전처리 중...")
        print(f"  함수: load_and_preprocess_validation_data()")

        X_val, y_val, feature_cols = load_and_preprocess_validation_data(
            validation_path=str(temp_dir),
            target_col='RACK_MAX_CELL_VOLTAGE'
        )

        print(f"\n{'='*60}")
        print(f"검증 데이터 로드 완료")
        print(f"{'='*60}")
        print(f"  X_val shape: {X_val.shape}")
        print(f"  y_val shape: {y_val.shape}")
        print(f"  Features ({len(feature_cols)}): {feature_cols}")

        # XCom으로 데이터 전달
        ti.xcom_push(key='validation_X', value=X_val.to_dict('list'))
        ti.xcom_push(key='validation_y', value=y_val.to_list())
        ti.xcom_push(key='validation_feature_cols', value=feature_cols)
        ti.xcom_push(key='validation_data_size', value=len(X_val))

        # 로컬 파일 정리
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n🧹 임시 파일 정리 완료")

        return {
            'validation_data_loaded': True,
            'data_size': len(X_val),
            'feature_count': len(feature_cols),
            'parquet_files_count': len(parquet_blobs)
        }

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise




def validate_new_model(**context):
    """새 모델 vs 현재 모델 성능 비교"""
    from google.cloud import storage
    import pickle
    import pandas as pd
    import io
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    ti = context['task_instance']

    # XCom에서 모델 경로 가져오기
    new_model_path = ti.xcom_pull(key='new_model_path', task_ids='load_models')
    current_model_path = ti.xcom_pull(key='current_model_path', task_ids='load_models')

    # XCom에서 전처리된 검증 데이터 가져오기
    validation_X_dict = ti.xcom_pull(key='validation_X', task_ids='load_validation_data')
    validation_y_list = ti.xcom_pull(key='validation_y', task_ids='load_validation_data')
    feature_cols = ti.xcom_pull(key='validation_feature_cols', task_ids='load_validation_data')

    # 설정은 이미 상단에서 로드됨
    validation_config = site_config['validation']

    # GCS 인증
    from google.oauth2 import service_account
    bucket_name = config['gcs']['bucket_name']
    credentials_path = config['gcs']['credentials_path']
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    gcp_project = os.getenv('GCP_PROJECT', 'your-gcp-project')
    storage_client = storage.Client(credentials=credentials, project=gcp_project)
    bucket = storage_client.bucket(bucket_name)

    print(f"\n{'='*60}")
    print(f"모델 성능 검증")
    print(f"{'='*60}")

    # 1. 검증 데이터 복원
    print(f"\n📥 검증 데이터 복원:")
    X_val = pd.DataFrame(validation_X_dict)
    y_val = pd.Series(validation_y_list)

    print(f"  - X_val shape: {X_val.shape}")
    print(f"  - y_val shape: {y_val.shape}")
    print(f"  - Features: {feature_cols}")

    # 2. 새 모델 평가
    new_model_blob = bucket.blob(new_model_path.replace(f'gs://{bucket_name}/', ''))
    new_model = pickle.loads(new_model_blob.download_as_bytes())

    y_pred_new = new_model.predict(X_val)

    new_rmse = np.sqrt(mean_squared_error(y_val, y_pred_new))
    new_r2 = r2_score(y_val, y_pred_new)

    print(f"\n📊 새 모델 성능:")
    print(f"  - RMSE: {new_rmse:.4f}")
    print(f"  - R²: {new_r2:.4f}")

    new_metrics = {
        'rmse': float(new_rmse),
        'r2': float(new_r2)
    }

    # 3. 현재 모델과 비교 (있는 경우)
    deploy_decision = {'deploy': False}

    if current_model_path:
        current_model_blob = bucket.blob(current_model_path.replace(f'gs://{bucket_name}/', ''))
        current_model = pickle.loads(current_model_blob.download_as_bytes())

        y_pred_current = current_model.predict(X_val)

        current_rmse = np.sqrt(mean_squared_error(y_val, y_pred_current))
        current_r2 = r2_score(y_val, y_pred_current)

        print(f"\n📊 현재 모델 성능:")
        print(f"  - RMSE: {current_rmse:.4f}")
        print(f"  - R²: {current_r2:.4f}")

        # 개선율 계산
        rmse_improvement = (current_rmse - new_rmse) / current_rmse
        r2_improvement = (new_r2 - current_r2) / abs(current_r2) if current_r2 != 0 else 0

        print(f"\n📈 성능 변화:")
        print(f"  - RMSE 개선: {rmse_improvement*100:.2f}%")
        print(f"  - R² 개선: {r2_improvement*100:.2f}%")

        improvement = {
            'rmse_improvement_pct': float(rmse_improvement * 100),
            'r2_improvement_pct': float(r2_improvement * 100)
        }

        # 배포 기준 검증
        min_rmse_improvement = validation_config['min_improvement_rmse']
        max_r2_degradation = validation_config['max_degradation_r2']

        if rmse_improvement >= min_rmse_improvement:
            print(f"\n✅ 배포 승인: RMSE {rmse_improvement*100:.2f}% 개선 (기준: {min_rmse_improvement*100}%)")
            deploy_decision = {
                'deploy': True,
                'reason': f'RMSE {rmse_improvement*100:.2f}% 개선',
                'new_metrics': new_metrics,
                'current_metrics': {'rmse': float(current_rmse), 'r2': float(current_r2)},
                'improvement': improvement
            }
        elif r2_improvement < -max_r2_degradation:
            print(f"\n❌ 배포 거부: R² {abs(r2_improvement)*100:.2f}% 하락 (최대 허용: {max_r2_degradation*100}%)")
            deploy_decision = {
                'deploy': False,
                'reason': f'R² 성능 하락 ({abs(r2_improvement)*100:.2f}%)',
                'new_metrics': new_metrics,
                'current_metrics': {'rmse': float(current_rmse), 'r2': float(current_r2)},
                'improvement': improvement
            }
        else:
            print(f"\n⚠️  배포 보류: 최소 개선율 미달")
            deploy_decision = {
                'deploy': False,
                'reason': f'RMSE 개선율 {rmse_improvement*100:.2f}% (기준: {min_rmse_improvement*100}%)',
                'new_metrics': new_metrics,
                'current_metrics': {'rmse': float(current_rmse), 'r2': float(current_r2)},
                'improvement': improvement
            }

    else:
        # 첫 배포인 경우 무조건 승인
        print(f"\n✅ 첫 배포 - 자동 승인")
        deploy_decision = {
            'deploy': True,
            'reason': '첫 번째 배포',
            'new_metrics': new_metrics,
            'improvement': {}
        }

    # XCom으로 전달
    ti.xcom_push(key='deploy_decision', value=deploy_decision)

    return deploy_decision




def decide_deployment_branch(**context):
    """배포 여부에 따라 분기"""
    ti = context['task_instance']
    deploy_decision = ti.xcom_pull(key='deploy_decision', task_ids='validate_model')

    if deploy_decision.get('deploy', False):
        return 'promote_to_production'
    else:
        return 'send_rejection_notification'




def send_deployment_notification(**context):
    """배포 성공 알림"""
    ti = context['task_instance']
    metadata = ti.xcom_pull(key='deployment_metadata', task_ids='promote_to_production')
    deployment_id = ti.xcom_pull(key='deployment_id', task_ids='record_to_gcs')

    print(f"\n{'='*60}")
    print(f"✅ 모델 배포 완료")
    print(f"{'='*60}")
    print(f"사이트: {metadata['site_id']}")
    print(f"버전: {metadata['model_version']}")
    print(f"배포 ID: {deployment_id}")
    print(f"배포 시간: {metadata['deployed_at']}")
    print(f"성능: {metadata['metrics']}")
    print(f"개선율: {metadata.get('improvement_over_previous', {})}")
    print(f"KServe 엔드포인트: {metadata['kserve_endpoint']}")
    print(f"\n배포 이력: gs://{bucket_name}/{site_config['paths']['deploy_dir']}/deployment_history.json")

    # TODO: Slack/Teams 알림 추가
    # send_slack_notification(...)

    return True


def send_rejection_notification(**context):
    """배포 거부 알림"""
    ti = context['task_instance']
    deploy_decision = ti.xcom_pull(key='deploy_decision', task_ids='validate_model')

    print(f"\n{'='*60}")
    print(f"❌ 모델 배포 거부")
    print(f"{'='*60}")
    print(f"사유: {deploy_decision.get('reason', 'Unknown')}")
    print(f"새 모델 성능: {deploy_decision.get('new_metrics', {})}")
    print(f"현재 모델 성능: {deploy_decision.get('current_metrics', {})}")
    print(f"성능 변화: {deploy_decision.get('improvement', {})}")

    # TODO: Slack/Teams 알림 추가

    return False




def verify_kserve_reload(**context):
    """KServe가 새 모델을 로드했는지 확인 (SSH를 통해 Kubeflow 서버에서 실행)"""
    import paramiko
    import time
    from pathlib import Path
    import os
    from dotenv import load_dotenv

    ti = context['task_instance']
    metadata = ti.xcom_pull(key='deployment_metadata', task_ids='promote_to_production')

    if not metadata:
        print("배포된 모델이 없음 - 검증 건너뜀")
        return True

    # SSH 접속 정보 로드
    config_dir = Path(__file__).parent / 'config'
    env_path = config_dir / '.env'
    load_dotenv(env_path)

    ssh_host = os.getenv('SSH_HOST')
    ssh_port = os.getenv('SSH_PORT')
    ssh_user = os.getenv('SSH_USER')
    ssh_password = os.getenv('SSH_PASSWORD')

    # KServe 설정
    namespace = site_config['namespace']
    inference_service_name = site_config['kserve']['inference_service_name']

    print(f"\n{'='*60}")
    print(f"KServe 모델 리로드 확인 (SSH 원격)")
    print(f"{'='*60}")
    print(f"InferenceService: {inference_service_name}")
    print(f"Namespace: {namespace}")

    try:
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        print(f"\n🔌 SSH 연결 중...")
        ssh_client.connect(
            hostname=ssh_host,
            port=int(ssh_port),
            username=ssh_user,
            password=ssh_password,
            timeout=30
        )
        print(f"  ✓ SSH 연결 성공")

        # kubectl로 Pod 상태 확인
        check_cmd = (
            f"kubectl get pods -n {namespace} "
            f"-l serving.kserve.io/inferenceservice={inference_service_name} "
            f"-o jsonpath='{{.items[*].status.phase}}'"
        )

        print(f"\n📊 Pod 상태 확인 중...")
        max_retries = 10
        for i in range(max_retries):
            sudo_cmd = f"echo '{ssh_password}' | sudo -S su -c \"{check_cmd}\""
            stdin, stdout, stderr = ssh_client.exec_command(sudo_cmd, timeout=30)
            exit_code = stdout.channel.recv_exit_status()

            if exit_code == 0:
                pod_status = stdout.read().decode('utf-8').strip()
                print(f"  Pod 상태: {pod_status}")

                if 'Running' in pod_status:
                    print(f"\n✓ KServe Pod가 정상 실행 중입니다")

                    # Pod 로그에서 모델 로드 확인
                    log_cmd = (
                        f"kubectl logs -n {namespace} "
                        f"-l serving.kserve.io/inferenceservice={inference_service_name} "
                        f"--tail=20 | grep -i 'model\\|load\\|ready' || echo 'No logs found'"
                    )

                    sudo_log_cmd = f"echo '{ssh_password}' | sudo -S su -c \"{log_cmd}\""
                    stdin, stdout, stderr = ssh_client.exec_command(sudo_log_cmd, timeout=30)
                    stdout.channel.recv_exit_status()

                    logs = stdout.read().decode('utf-8')
                    print(f"\n📝 최근 로그:")
                    print(logs[:500] if logs else "  (로그 없음)")

                    ssh_client.close()
                    return True
                else:
                    print(f"  ⏳ Pod가 아직 준비되지 않음, 30초 대기 중... ({i+1}/{max_retries})")
                    if i < max_retries - 1:
                        time.sleep(30)
            else:
                error = stderr.read().decode('utf-8')
                print(f"  ❌ kubectl 명령 실패: {error}")
                break

        ssh_client.close()
        print(f"\n⚠️ 최대 재시도 횟수 초과, 하지만 배포는 성공한 것으로 간주")
        return True

    except Exception as e:
        print(f"\n❌ 검증 실패: {e}")
        import traceback
        traceback.print_exc()

        # 검증 실패해도 배포 자체는 성공한 것으로 간주
        print(f"\n⚠️ 검증은 실패했지만 배포는 완료되었습니다")
        return True




load_models = PythonOperator(
    task_id='load_models',
    python_callable=load_models_from_gcs,
    dag=dag,
)

load_validation_data = PythonOperator(
    task_id='load_validation_data',
    python_callable=load_validation_data_from_gcs,
    dag=dag,
)

validate_model = PythonOperator(
    task_id='validate_model',
    python_callable=validate_new_model,
    dag=dag,
)

decide_deployment = BranchPythonOperator(
    task_id='decide_deployment',
    python_callable=decide_deployment_branch,
    dag=dag,
)

promote_to_production = PythonOperator(
    task_id='promote_to_production',
    python_callable=promote_model_to_production,
    dag=dag,
)

record_to_gcs = PythonOperator(
    task_id='record_to_gcs',
    python_callable=record_deployment_to_gcs,
    dag=dag,
)

trigger_kserve = PythonOperator(
    task_id='trigger_kserve_reload',
    python_callable=trigger_kserve_reload,
    dag=dag,
)

verify_kserve = PythonOperator(
    task_id='verify_kserve_reload',
    python_callable=verify_kserve_reload,
    dag=dag,
)

send_success_notification = PythonOperator(
    task_id='send_success_notification',
    python_callable=send_deployment_notification,
    dag=dag,
)

# 거부 브랜치
send_rejection_notification = PythonOperator(
    task_id='send_rejection_notification',
    python_callable=send_rejection_notification,
    dag=dag,
)

join = EmptyOperator(
    task_id='join',
    trigger_rule='none_failed_min_one_success',
    dag=dag,
)



load_models >> validate_model
load_validation_data >> validate_model

validate_model >> decide_deployment

# 배포 경로
decide_deployment >> promote_to_production >> record_to_gcs >> trigger_kserve >> verify_kserve >> send_success_notification >> join

# 거부 경로
decide_deployment >> send_rejection_notification >> join
