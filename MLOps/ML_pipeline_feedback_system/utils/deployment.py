"""KServe 모델 배포 유틸리티"""

import yaml
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from google.cloud import storage


class DeploymentConfig:
    def __init__(self, config_path: str = None):
        if config_path is None:
            current_dir = Path(__file__).parent.parent
            config_path = current_dir / 'config' / 'deploy.yaml'

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def get_site_config(self, site_id: str) -> Dict[str, Any]:
        if site_id not in self.config['sites']:
            raise ValueError(f"Site '{site_id}' not found in deploy.yaml")
        return self.config['sites'][site_id]

    def get_gcs_bucket(self) -> str:
        return self.config['gcs']['bucket_name']


class ModelDeployment:
    def __init__(self, site_id: str, config_path: str = None):
        self.site_id = site_id
        self.config = DeploymentConfig(config_path) if config_path else DeploymentConfig()
        self.site_config = self.config.get_site_config(site_id)
        self.bucket_name = self.config.get_gcs_bucket()

        # GCS 클라이언트 생성
        from google.oauth2 import service_account
        import os

        # credentials_path를 config에서 읽거나 환경변수 사용
        credentials_path = self.config.config['gcs'].get('credentials_path')
        if not credentials_path:
            credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '/path/to/credentials.json')

        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        gcp_project = os.getenv('GCP_PROJECT', 'your-gcp-project')
        self.storage_client = storage.Client(credentials=credentials, project=gcp_project)
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def promote_to_production(self, model_version: str, deploy_decision: Dict[str, Any]) -> Dict[str, Any]:
        if not deploy_decision.get('deploy', False):
            print("❌ 배포 조건 미충족 - 건너뜀")
            return None

        paths = self.site_config['paths']
        deploy_files = self.site_config['deploy_files']

        source_model_path = f"{paths['models_dir']}/{model_version}/{model_version}_xgboost_{self.site_id}_model.pkl"
        source_params_path = f"{paths['models_dir']}/{model_version}/{model_version}_xgboost_{self.site_id}.json"

        deploy_model_path = f"{paths['deploy_dir']}/{deploy_files['model']}"
        deploy_params_path = f"{paths['deploy_dir']}/{deploy_files['hyperparameters']}"
        deploy_metadata_path = f"{paths['deploy_dir']}/{deploy_files['metadata']}"

        print(f"\n{'='*60}")
        print(f"모델 배포 시작: {self.site_id}")
        print(f"{'='*60}")
        print(f"\n🚀 새 모델 배포 중...")
        print(f"  원본: gs://{self.bucket_name}/{source_model_path}")
        print(f"  대상: gs://{self.bucket_name}/{deploy_model_path}")

        source_model_blob = self.bucket.blob(source_model_path)
        self.bucket.copy_blob(source_model_blob, self.bucket, deploy_model_path)

        source_params_blob = self.bucket.blob(source_params_path)
        self.bucket.copy_blob(source_params_blob, self.bucket, deploy_params_path)

        metadata = {
            'site_id': self.site_id,
            'model_version': model_version,
            'deployed_at': datetime.now().isoformat(),
            'source_model_path': f'gs://{self.bucket_name}/{source_model_path}',
            'deploy_model_path': f'gs://{self.bucket_name}/{deploy_model_path}',
            'metrics': deploy_decision.get('new_metrics', {}),
            'improvement_over_previous': deploy_decision.get('improvement', {}),
            'deployment_method': 'automatic',
            'kserve_endpoint': self.site_config['kserve']['endpoint'],
            'namespace': self.site_config['namespace']
        }

        metadata_blob = self.bucket.blob(deploy_metadata_path)
        metadata_blob.upload_from_string(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            content_type='application/json'
        )

        print(f"  ✓ 모델 배포 완료!")
        print(f"\n📊 배포 정보:")
        print(f"  - 모델 버전: {model_version}")
        print(f"  - 배포 시간: {metadata['deployed_at']}")

        return metadata

    def get_deployment_history_from_gcs(self) -> Dict[str, Any]:
        paths = self.site_config['paths']
        deploy_files = self.site_config['deploy_files']

        history_path = f"{paths['deploy_dir']}/{deploy_files['deployment_history']}"
        history_blob = self.bucket.blob(history_path)

        if history_blob.exists():
            return json.loads(history_blob.download_as_string())
        else:
            return {
                'site_id': self.site_id,
                'created_at': datetime.now().isoformat(),
                'deployments': []
            }

    def save_deployment_history_to_gcs(self, history: Dict[str, Any]) -> None:
        paths = self.site_config['paths']
        deploy_files = self.site_config['deploy_files']

        history_path = f"{paths['deploy_dir']}/{deploy_files['deployment_history']}"
        history_blob = self.bucket.blob(history_path)

        history['last_updated'] = datetime.now().isoformat()
        history['total_deployments'] = len(history['deployments'])

        history_blob.upload_from_string(
            json.dumps(history, indent=2, ensure_ascii=False),
            content_type='application/json'
        )

        print(f"\n📝 배포 이력 저장 완료: gs://{self.bucket_name}/{history_path}")

    def record_deployment(self, metadata: Dict[str, Any]) -> str:
        history = self.get_deployment_history_from_gcs()

        deployment_record = {
            'deployment_id': f"{self.site_id}_{metadata['model_version']}_{int(datetime.now().timestamp())}",
            'model_version': metadata['model_version'],
            'deployed_at': metadata['deployed_at'],
            'source_model_path': metadata['source_model_path'],
            'metrics': metadata['metrics'],
            'improvement': metadata.get('improvement_over_previous', {}),
            'deployment_method': metadata['deployment_method'],
            'status': 'active',
            'kserve_endpoint': metadata['kserve_endpoint']
        }

        history['deployments'].insert(0, deployment_record)
        self.save_deployment_history_to_gcs(history)

        print(f"\n✅ 배포 이력 기록 완료 - {deployment_record['deployment_id']}")

        return deployment_record['deployment_id']


def promote_model_to_production(**context):
    import os
    ti = context['task_instance']
    execution_date = context['execution_date']

    site_id = context['dag'].tags[0] if context['dag'].tags else os.getenv('SITE_ID', 'default_site')
    model_version = execution_date.strftime('%Y%m')

    deploy_decision = ti.xcom_pull(key='deploy_decision', task_ids='validate_model')

    deployment = ModelDeployment(site_id)
    metadata = deployment.promote_to_production(model_version, deploy_decision)

    if metadata:
        ti.xcom_push(key='deployment_metadata', value=metadata)
        return metadata
    else:
        return None


def record_deployment_to_gcs(**context):
    ti = context['task_instance']
    metadata = ti.xcom_pull(key='deployment_metadata', task_ids='promote_to_production')

    if not metadata:
        print("배포된 모델이 없음 - 배포 이력 기록 건너뜀")
        return None

    deployment = ModelDeployment(metadata['site_id'])
    deployment_id = deployment.record_deployment(metadata)

    ti.xcom_push(key='deployment_id', value=deployment_id)
    return {'deployment_id': deployment_id}


def trigger_kserve_reload(**context):
    ti = context['task_instance']
    metadata = ti.xcom_pull(key='deployment_metadata', task_ids='promote_to_production')

    if not metadata:
        print("배포된 모델 없음 - KServe 업데이트 건너뜀")
        return False

    site_id = metadata['site_id']
    model_version = metadata['model_version']
    namespace = metadata['namespace']

    config = DeploymentConfig()
    site_config = config.get_site_config(site_id)
    inference_service_name = site_config['kserve']['inference_service_name']

    print(f"\n{'='*60}")
    print(f"KServe InferenceService 업데이트 (SSH 원격 실행)")
    print(f"{'='*60}")
    print(f"  Service: {inference_service_name}")
    print(f"  Namespace: {namespace}")
    print(f"  모델 버전: {model_version}")

    # SSH 접속 정보 로드 (.env 파일에서)
    from pathlib import Path
    import os
    from dotenv import load_dotenv

    config_dir = Path(__file__).parent.parent / 'config'
    env_path = config_dir / '.env'
    load_dotenv(env_path)

    ssh_host = os.getenv('SSH_HOST')
    ssh_port = os.getenv('SSH_PORT')
    ssh_user = os.getenv('SSH_USER')
    ssh_password = os.getenv('SSH_PASSWORD')

    if not all([ssh_host, ssh_port, ssh_user, ssh_password]):
        raise ValueError("SSH 접속 정보가 .env 파일에 없습니다 (SSH_HOST, SSH_PORT, SSH_USER, SSH_PASSWORD)")

    print(f"\n🔐 SSH 연결 정보:")
    print(f"  Host: {ssh_user}@{ssh_host}:{ssh_port}")

    annotations = [
        f'deployment.info/model-version={model_version}',
        f'deployment.info/last-updated={datetime.now().isoformat()}',
        f'deployment.info/source-path={metadata["source_model_path"]}'
    ]

    # kubectl 명령어 생성 (sudo su 후 실행)
    kubectl_annotate_cmd = (
        f"kubectl annotate inferenceservice {inference_service_name} "
        f"-n {namespace} --overwrite "
        f"{' '.join(annotations)}"
    )

    kubectl_wait_cmd = (
        f"kubectl rollout status deployment "
        f"-n {namespace} "
        f"-l serving.kserve.io/inferenceservice={inference_service_name} "
        f"--timeout=5m"
    )

    kubectl_restart_cmd = (
        f"kubectl rollout restart deployment "
        f"-n {namespace} "
        f"-l serving.kserve.io/inferenceservice={inference_service_name}"
    )

    # paramiko import를 함수 시작 부분으로 이동
    import paramiko

    def run_sudo_command(ssh_client, command, timeout=60):
        """
        sudo su 후 kubectl 명령어 실행

        Args:
            ssh_client: paramiko SSH 클라이언트
            command: 실행할 kubectl 명령어
            timeout: 타임아웃 (초)

        Returns:
            tuple: (exit_code, stdout, stderr)
        """
        # sudo su -c 를 사용하여 root 권한으로 명령 실행
        # 비밀번호는 echo로 전달
        sudo_cmd = f"echo '{ssh_password}' | sudo -S su -c \"{command}\""

        stdin, stdout, stderr = ssh_client.exec_command(sudo_cmd, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        stdout_text = stdout.read().decode('utf-8')
        stderr_text = stderr.read().decode('utf-8')

        # sudo 비밀번호 프롬프트 제거
        stderr_text = '\n'.join([
            line for line in stderr_text.split('\n')
            if 'sudo' not in line.lower() and 'password' not in line.lower()
        ]).strip()

        return exit_code, stdout_text, stderr_text

    try:
        # SSH 클라이언트 생성
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

        # 1. Annotation 업데이트
        print(f"\n📝 Annotation 업데이트 중 (sudo su)...")
        print(f"  명령어: {kubectl_annotate_cmd}")

        exit_code, stdout_text, stderr_text = run_sudo_command(
            ssh_client, kubectl_annotate_cmd, timeout=60
        )

        if exit_code == 0:
            print(f"  ✓ Annotation 업데이트 완료")
            if stdout_text:
                print(stdout_text)
        else:
            print(f"  ❌ Annotation 업데이트 실패: {stderr_text}")
            raise Exception(f"kubectl annotate 실패: {stderr_text}")

        # 2. Rolling Update 대기
        print(f"\n⏳ Rolling update 대기 중 (최대 5분, sudo su)...")

        exit_code, stdout_text, stderr_text = run_sudo_command(
            ssh_client, kubectl_wait_cmd, timeout=320
        )

        if exit_code == 0:
            print(f"  ✓ Rolling update 완료")
            if stdout_text:
                print(stdout_text)
            ssh_client.close()

            ti.xcom_push(key='kserve_reload_success', value=True)
            return True
        else:
            print(f"  ⚠️ Rolling update 대기 실패, Pod 재시작 시도")
            print(f"  오류: {stderr_text}")

    except Exception as e:
        print(f"\n⚠️ Annotation 방식 실패: {e}")
        print(f"  대체 방법 시도: Pod 재시작")

    # 대체 방법: Pod 재시작
    try:
        if 'ssh_client' not in locals() or ssh_client.get_transport() is None:
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_client.connect(
                hostname=ssh_host,
                port=int(ssh_port),
                username=ssh_user,
                password=ssh_password,
                timeout=30
            )

        print(f"\n🔄 Pod 재시작 중 (sudo su)...")
        print(f"  명령어: {kubectl_restart_cmd}")

        exit_code, stdout_text, stderr_text = run_sudo_command(
            ssh_client, kubectl_restart_cmd, timeout=60
        )

        if exit_code == 0:
            print(f"  ✓ Pod 재시작 완료")
            if stdout_text:
                print(stdout_text)
            ssh_client.close()

            ti.xcom_push(key='kserve_reload_success', value=True)
            return True
        else:
            print(f"  ❌ Pod 재시작 실패: {stderr_text}")
            ssh_client.close()

            ti.xcom_push(key='kserve_reload_success', value=False)
            raise Exception(f"kubectl rollout restart 실패: {stderr_text}")

    except Exception as e:
        print(f"❌ 모든 방법 실패: {e}")
        if 'ssh_client' in locals():
            ssh_client.close()

        ti.xcom_push(key='kserve_reload_success', value=False)
        raise


def get_deployment_history(site_id: str = 'default_site', limit: int = 10):
    deployment = ModelDeployment(site_id)
    history = deployment.get_deployment_history_from_gcs()

    print(f"\n{'='*60}")
    print(f"{site_id} 배포 이력")
    print(f"{'='*60}")
    print(f"총 배포 횟수: {history.get('total_deployments', 0)}\n")

    for i, record in enumerate(history['deployments'][:limit], 1):
        print(f"{i}. 버전: {record['model_version']}")
        print(f"   배포 시간: {record['deployed_at']}")
        print(f"   성능: {record['metrics']}")
        print()

    return history
