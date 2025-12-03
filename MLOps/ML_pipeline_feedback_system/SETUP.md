# 환경 설정 가이드

이 프로젝트를 사용하기 전에 아래의 환경 변수들을 설정해야 합니다.

## 1. 환경변수 설정

`config/.env.example` 파일을 복사하여 `config/.env` 파일을 생성하고 실제 값으로 수정하세요.

```bash
cp config/.env.example config/.env
```

### 필수 환경변수

#### Kubeflow 설정
- `HOST`: Kubeflow 호스트 URL (예: https://kubeflow.example.com)
- `USERNAME`: Kubeflow 인증 사용자명
- `PASSWORD`: Kubeflow 인증 비밀번호
- `NAMESPACE`: Kubernetes 네임스페이스

#### 사이트 설정
- `SITE_ID`: 사이트 고유 식별자 (예: site1, site2)

#### GCS 설정
- `GCS_BUCKET`: Google Cloud Storage 버킷명
- `GCS_MODEL_BASE_PATH`: 모델 저장 기본 경로 (기본값: vt-model)

#### UI 설정
- `KUBEFLOW_UI`: Kubeflow UI URL

## 2. DAG 파일 수정

`dag_katib_tuning.py` 파일에서 환경변수를 사용하도록 설정되어 있습니다.
추가로 사이트별 설정이 필요한 경우 `config/katib_config.yaml` 파일을 수정하세요.

## 3. Kubernetes Secret 설정

GCP 인증을 위한 Secret을 생성해야 합니다:

```bash
kubectl create secret generic gcp-credentials \
  --from-file=key.json=/path/to/your/credential.json \
  -n your-namespace
```

## 4. 주요 변경사항

### 제거된 하드코딩 값들
- 사이트 ID 'panly' → 환경변수 `SITE_ID`
- GCS 버킷명 → 환경변수 `GCS_BUCKET`
- Kubeflow URL → 환경변수 `KUBEFLOW_UI`
- 네임스페이스 → 환경변수 `NAMESPACE`

### 일반화된 경로
- 데이터 경로: `/mnt/ess-dataset/{SITE_ID}/data`
- 모델 저장 경로: `gs://{GCS_BUCKET}/{GCS_MODEL_BASE_PATH}/{SITE_ID}/models/yyyymm/`

## 5. 사용 예시

### Airflow에서 사용
```python
# 환경변수가 설정되어 있으면 자동으로 사용됩니다
import os
os.environ['SITE_ID'] = 'my-site'
os.environ['GCS_BUCKET'] = 'my-bucket'
```

### Docker 컨테이너에서 사용
```bash
docker run -e SITE_ID=my-site \
           -e GCS_BUCKET=my-bucket \
           -e HOST=https://kubeflow.example.com \
           your-image:tag
```

## 주의사항

- `.env` 파일은 절대 Git에 커밋하지 마세요
- `.gitignore`에 `.env` 파일이 포함되어 있는지 확인하세요
- 민감한 정보(비밀번호, 인증 정보)는 반드시 환경변수나 Secret으로 관리하세요
