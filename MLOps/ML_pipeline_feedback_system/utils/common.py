"""
공통 유틸리티 함수 모듈
여러 DAG에서 공통으로 사용하는 함수들을 제공합니다.
"""
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


def get_gcs_client(credential_path: str = None, project_id: str = None):
    """
    GCS 클라이언트 초기화 (중복 코드 제거용 헬퍼)

    Args:
        credential_path: GCP 인증 파일 경로 (환경변수 GCP_CREDENTIAL_PATH 또는 설정 파일 사용)
        project_id: GCP 프로젝트 ID (환경변수 GCP_PROJECT_ID 또는 설정 파일 사용)

    Returns:
        google.cloud.storage.Client 객체
    """
    from google.cloud import storage
    from google.oauth2 import service_account

    # 환경 변수에서 기본값 가져오기
    if credential_path is None:
        credential_path = os.getenv('GCP_CREDENTIAL_PATH')
        if credential_path is None:
            raise ValueError("GCP_CREDENTIAL_PATH 환경 변수 또는 credential_path 파라미터를 설정해야 합니다")

    if project_id is None:
        project_id = os.getenv('GCP_PROJECT_ID')

    credentials = service_account.Credentials.from_service_account_file(credential_path)

    # project_id가 없으면 credential 파일에서 자동으로 가져옴
    if project_id:
        return storage.Client(credentials=credentials, project=project_id)
    else:
        return storage.Client(credentials=credentials, project=credentials.project_id)


def get_site_credential(site_config) -> str:
    """
    사이트의 GCP 인증 정보 가져오기

    Args:
        site_config: 사이트 설정 딕셔너리

    Returns:
        인증 JSON 문자열
    """
    credential_path = site_config['gcp']['credential_path']

    if not os.path.exists(credential_path):
        raise FileNotFoundError(f"인증 파일을 찾을 수 없습니다: {credential_path}")

    with open(credential_path, 'r') as f:
        return f.read()


def ensure_directory_exists(directory: str):
    """
    디렉토리가 존재하지 않으면 생성

    Args:
        directory: 디렉토리 경로
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_date_range_for_training(
    execution_date: datetime,
    training_days: int = 90
) -> tuple:
    """
    학습에 사용할 날짜 범위 계산

    Args:
        execution_date: Airflow 실행 날짜
        training_days: 학습에 사용할 데이터 기간 (일)

    Returns:
        (start_date, end_date) 튜플
    """
    end_date = execution_date - timedelta(days=1)
    start_date = end_date - timedelta(days=training_days)
    return start_date, end_date


def format_pipeline_params(
    site_config,
    site_id: str,
    start_date: datetime,
    end_date: datetime,
    model_version: str,
    best_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Kubeflow 파이프라인 파라미터 포맷팅

    Args:
        site_config: 사이트 설정 딕셔너리
        site_id: 사이트 ID
        start_date: 시작 날짜
        end_date: 종료 날짜
        model_version: 모델 버전
        best_params: 최적 하이퍼파라미터 (선택적)

    Returns:
        파이프라인 파라미터 딕셔너리
    """
    credential_json = get_site_credential(site_config)

    pipeline_params = {
        'bucket_name': site_config['gcp']['bucket_name'],
        'credential_json': credential_json,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'model_version': model_version,
        # 모델 저장 GCS 설정
        'model_storage_bucket': site_config.get('model_storage', {}).get('bucket_name', os.getenv('MODEL_STORAGE_BUCKET', 'model-storage-bucket')),
        'model_base_path': site_config.get('model_storage', {}).get('base_path', f'vt-model/{site_id}'),
    }

    # 하이퍼파라미터 추가
    if best_params:
        pipeline_params.update({
            'xgb_learning_rate': float(best_params.get('xgb_learning_rate', 0.1)),
            'xgb_max_depth': int(best_params.get('xgb_max_depth', 4)),
            'xgb_n_estimators': int(best_params.get('xgb_n_estimators', 2000)),
            'xgb_subsample': float(best_params.get('xgb_subsample', 0.85)),
        })
    else:
        # 기본 하이퍼파라미터 사용 (YAML에서 로드)
        xgb_params = site_config.get('models', {}).get('xgboost', {})

        pipeline_params.update({
            'xgb_learning_rate': xgb_params.get('learning_rate', 0.1),
            'xgb_max_depth': xgb_params.get('max_depth', 4),
            'xgb_n_estimators': xgb_params.get('n_estimators', 2000),
            'xgb_subsample': xgb_params.get('subsample', 0.85),
        })

    return pipeline_params


def cleanup_old_files(
    directory: str,
    retention_days: int,
    pattern: str = '*.parquet'
) -> List[str]:
    """
    보관 기간이 지난 파일 삭제

    Args:
        directory: 디렉토리 경로
        retention_days: 보관 기간 (일)
        pattern: 파일 패턴 (기본값: '*.parquet')

    Returns:
        삭제된 파일 이름 리스트
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    cutoff_date = datetime.now() - timedelta(days=retention_days)
    removed_files = []

    for file_path in dir_path.glob(pattern):
        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)

        if file_time < cutoff_date:
            print(f"  오래된 파일 삭제: {file_path.name}")
            file_path.unlink()
            removed_files.append(file_path.name)

    return removed_files


def get_default_hyperparameters() -> Dict[str, Any]:
    """
    기본 하이퍼파라미터 가져오기 (Katib 실패 시 사용)

    Returns:
        기본 하이퍼파라미터 딕셔너리 (XGBoost만 사용)
    """
    return {
        'xgb_learning_rate': 0.1,
        'xgb_max_depth': 4,
        'xgb_n_estimators': 2000,
        'xgb_subsample': 0.85,
    }


def format_completion_message(
    execution_date: datetime,
    run_id: str,
    best_params: Optional[Dict[str, Any]],
    validation_results: Optional[Dict[str, Any]]
) -> str:
    """
    학습 완료 메시지 포맷팅

    Args:
        execution_date: 실행 날짜
        run_id: 파이프라인 실행 ID
        best_params: 최적 하이퍼파라미터
        validation_results: 검증 결과

    Returns:
        포맷된 메시지 문자열
    """
    import json

    message = f"""
    ============================================================
    월별 배터리 모델 학습 완료
    ============================================================

    날짜: {execution_date.strftime('%Y-%m-%d')}
    실행 ID: {run_id}

    최적 하이퍼파라미터:
    {json.dumps(best_params, indent=2) if best_params else 'N/A'}

    학습된 모델:
    {json.dumps(validation_results.get('models_trained', []), indent=2) if validation_results else 'N/A'}

    검증 상태: {validation_results.get('status', 'unknown') if validation_results else 'unknown'}

    ============================================================
    """

    return message


def get_trained_model_list() -> List[str]:
    """
    학습되는 모델 리스트 가져오기 (XGBoost만 사용)

    Returns:
        모델명 리스트 (1개: Max Voltage)
    """
    return [
        'max_voltage_xgb'
    ]


def validate_site_config(site_config) -> bool:
    """
    사이트 설정 유효성 검사

    Args:
        site_config: 사이트 설정 딕셔너리

    Returns:
        유효성 여부

    Raises:
        ValueError: 설정이 유효하지 않은 경우
    """
    # 인증 파일 확인
    credential_path = site_config['gcp']['credential_path']
    if not os.path.exists(credential_path):
        raise ValueError(
            f"GCP 인증 파일을 찾을 수 없습니다: {credential_path}"
        )

    # 데이터 디렉토리 생성 가능 여부 확인
    data_dir = site_config['paths']['data_dir']
    try:
        ensure_directory_exists(data_dir)
    except Exception as e:
        raise ValueError(
            f"데이터 디렉토리를 생성할 수 없습니다: {data_dir}\n{e}"
        )

    return True


# 사용 예시
if __name__ == '__main__':
    import yaml
    from pathlib import Path

    # 주석: 각 DAG가 별도의 config 파일을 사용하므로 테스트 시 적절한 config 파일 선택 필요
    # data_download_config.yaml 또는 katib_config.yaml
    config_path = Path(__file__).parent.parent / 'config' / 'katib_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 첫 번째 사이트 설정 가져오기
    site_id = list(config['sites'].keys())[0]
    site = config['sites'][site_id]

    # 설정 유효성 검사
    print("=== 설정 유효성 검사 ===")
    try:
        validate_site_config(site)
        print("✓ 설정이 유효합니다")
    except ValueError as e:
        print(f"✗ 설정 오류: {e}")

    # 날짜 범위 계산
    print("\n=== 학습 날짜 범위 ===")
    execution_date = datetime.now()
    start, end = get_date_range_for_training(execution_date, training_days=90)
    print(f"시작: {start.strftime('%Y-%m-%d')}")
    print(f"종료: {end.strftime('%Y-%m-%d')}")

    # 파이프라인 파라미터
    print("\n=== 파이프라인 파라미터 ===")
    params = format_pipeline_params(site, site_id, start, end, 'v202501')
    for key in ['bucket_name', 'start_date', 'end_date', 'model_version']:
        print(f"{key}: {params[key]}")


def load_best_params_from_gcs(
    site_config: Dict[str, Any],
    site_id: str,
    target_column: str = 'RACK_MAX_CELL_VOLTAGE'
) -> Optional[Dict[str, Any]]:
    """
    GCS에서 최적 하이퍼파라미터 로드

    Args:
        site_config: 사이트 설정 딕셔너리
        site_id: 사이트 ID
        target_column: 타겟 컬럼명 (기본값: 'RACK_MAX_CELL_VOLTAGE')

    Returns:
        최적 하이퍼파라미터 딕셔너리 (로드 실패 시 None)

    Example:
        >>> best_params = load_best_params_from_gcs(site_config, 'site_a')
        >>> if best_params:
        >>>     print(f"Learning rate: {best_params['xgb_learning_rate']}")
    """
    import json
    import tempfile
    from google.cloud import storage
    from google.oauth2 import service_account

    try:
        model_storage = site_config.get('model_storage', {})
        gcs_bucket = model_storage.get('bucket_name', os.getenv('MODEL_STORAGE_BUCKET', 'model-storage-bucket'))
        gcs_base_path = f"{model_storage.get('base_path', f'vt-model/{site_id}')}/katib"

        # GCS 클라이언트 설정
        credential_json = get_site_credential(site_config)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(credential_json)
            cred_path = f.name

        credentials = service_account.Credentials.from_service_account_file(cred_path)
        client = storage.Client(credentials=credentials, project=credentials.project_id)
        bucket = client.bucket(gcs_bucket)

        # 최신 파라미터 파일 다운로드
        blob_path = f"{gcs_base_path}/best_params_{target_column}_latest.json"
        blob = bucket.blob(blob_path)

        if not blob.exists():
            print(f"⚠️ GCS에 최적 파라미터가 없습니다: gs://{gcs_bucket}/{blob_path}")
            return None

        best_params_data = json.loads(blob.download_as_string())

        # Katib 파라미터 형식을 format_pipeline_params 형식으로 변환
        # Katib: {'xgb_learning_rate': 0.1, 'xgb_max_depth': 4, ...}
        # 그대로 사용 가능
        return best_params_data['parameters']

    except Exception as e:
        print(f"⚠️ 최적 파라미터 로드 실패: {e}")
        return None
