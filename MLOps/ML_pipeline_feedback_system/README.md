# ML Pipeline Feedback System

배터리 ESS(Energy Storage System) 예측 모델을 위한 자동화된 ML 파이프라인 시스템입니다. Katib를 활용한 하이퍼파라미터 튜닝, 모델 검증, 그리고 KServe를 통한 자동 배포를 지원합니다.

## 📋 목차

- [주요 기능](#주요-기능)
- [시스템 아키텍처](#시스템-아키텍처)
- [설치 및 설정](#설치-및-설정)
- [사용 방법](#사용-방법)
- [DAG 구성](#dag-구성)
- [디렉토리 구조](#디렉토리-구조)
- [환경 변수](#환경-변수)
- [트러블슈팅](#트러블슈팅)

## 🚀 주요 기능

### 1. 자동 하이퍼파라미터 튜닝 (Katib)
- **Bayesian Optimization** 기반 효율적인 탐색
- **Early Stopping** 지원으로 리소스 절약
- XGBoost 모델의 7개 주요 하이퍼파라미터 자동 튜닝
- GCS(Google Cloud Storage)에 베스트 모델 및 파라미터 자동 저장

### 2. 모델 검증 및 자동 배포
- 새 모델과 기존 모델 성능 자동 비교
- 설정 가능한 배포 임계값 (RMSE 개선율, R² 저하율)
- KServe를 통한 무중단 모델 배포
- 배포 이력 자동 기록 및 추적

### 3. 데이터 관리
- 일일 자동 데이터 다운로드 및 동기화
- GCS 기반 데이터 저장소
- Kubernetes CronJob을 통한 스케줄링

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Airflow (DAG 스케줄러)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │ 데이터 다운로드   │    │  Katib 튜닝 DAG  │    │ 모델 검증/배포    │  │
│  │ DAG (일 1회)     │    │  (월 1회)        │───▶│ DAG (튜닝 후)    │  │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
│           │                       │                        │             │
└───────────┼───────────────────────┼────────────────────────┼─────────────┘
            │                       │                        │
            ▼                       ▼                        ▼
   ┌─────────────────┐    ┌────────────────┐      ┌─────────────────┐
   │  Google Cloud   │    │  Kubeflow      │      │  KServe         │
   │  Storage (GCS)  │◀───│  (Katib)       │      │  (Inference)    │
   │                 │    └────────────────┘      └─────────────────┘
   │  - 원천 데이터   │             │                        │
   │  - 학습 데이터   │             └────────────────────────┘
   │  - 모델 저장소   │                        │
   └─────────────────┘◀───────────────────────┘
            ▲
            │
   ┌────────────────┐
   │  Kubernetes    │
   │  CronJob       │
   │  (데이터 동기화)│
   └────────────────┘
```

## 💾 설치 및 설정

### 사전 요구사항

- **Kubernetes Cluster** (1.21+)
- **Kubeflow** with Katib (1.7+)
- **KServe** (0.10+)
- **Apache Airflow** (2.5+)
- **Google Cloud Storage** 계정
- **Python** 3.12+

### 1. 레포지토리 클론

```bash
git clone https://github.com/your-org/ML_pipeline_feedback_system.git
cd ML_pipeline_feedback_system
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp config/.env.example config/.env

# .env 파일 편집
vim config/.env
```

필수 환경 변수:
```bash
# Kubeflow 설정
HOST=https://your-kubeflow-host
USERNAME=your-username
PASSWORD=your-password
NAMESPACE=your-namespace

# 사이트 설정
SITE_ID=your-site-id

# GCS 설정
GCS_BUCKET=your-gcs-bucket
GCS_MODEL_BASE_PATH=vt-model
GCP_PROJECT=your-gcp-project

# Kubeflow UI
KUBEFLOW_UI=https://your-kubeflow-url
```

### 3. YAML 설정 파일 수정

#### config/katib_config.yaml
```yaml
sites:
  your_site_id:  # SITE_ID와 일치해야 함
    name: "사이트명"
    description: "사이트 설명"

defaults:
  kubeflow:
    namespace: "your-namespace"
    training_image: "your-training-image:tag"
    max_trial_count: 12
    parallel_trial_count: 3
```

#### config/deploy.yaml
```yaml
gcs:
  bucket_name: your-gcs-bucket
  credentials_path: /path/to/credentials.json

sites:
  your_site_id:
    site_id: your_site_id
    namespace: your-namespace
    paths:
      models_dir: vt-model/your_site_id/models
      deploy_dir: vt-model/your_site_id/model-deploy
```

### 4. Kubernetes Secret 생성

```bash
# GCP 인증 Secret 생성
kubectl create secret generic gcp-credentials \
  --from-file=key.json=/path/to/your/gcp-credentials.json \
  -n your-namespace
```

### 5. Python 패키지 설치

```bash
pip install -r requirements.txt
```

## 📖 사용 방법

### Airflow에서 DAG 실행

1. **데이터 다운로드** (매일 오전 2시 자동 실행)
   ```bash
   # 수동 실행
   airflow dags trigger daily_data_download_your_site_id
   ```

2. **Katib 하이퍼파라미터 튜닝** (매월 1일 오전 3시 자동 실행)
   ```bash
   # 수동 실행
   airflow dags trigger katib_tuning_your_site_id
   ```

3. **모델 검증 및 배포** (Katib 완료 후 자동 트리거)
   ```bash
   # 수동 실행
   airflow dags trigger model_validation_deployment_your_site_id
   ```

### 로컬에서 개별 컴포넌트 테스트

#### Kubeflow 클라이언트 테스트
```bash
python utils/kubeflow_client.py
```

#### 배포 유틸리티 테스트
```bash
python utils/deployment.py
```

## 📂 DAG 구성

### 1. 데이터 다운로드 DAG (`dag_daily_data_download.py`)

**스케줄**: 매일 오전 2시

**주요 Task**:
1. `download_data`: GCP에서 원천 데이터 다운로드
2. `sync_to_gcs`: 처리된 데이터를 GCS에 동기화
3. `validate_data`: 데이터 무결성 검증
4. `cleanup_old_files`: 오래된 파일 정리 (설정된 retention 기준)

**데이터 타입**:
- `rack`: 배터리 랙 데이터
- `bank`: 배터리 뱅크 데이터
- `pcs`: PCS(Power Conversion System) 데이터
- `etc`: 기타 데이터

**출력**:
- `gs://{bucket}/{site_id}/data/rack/{date}.parquet`
- 로컬: `/path/to/data/{site_id}/rack/{date}.parquet`

### 2. Katib 튜닝 DAG (`dag_katib_tuning.py`)

**스케줄**: 매월 1일 오전 3시

**주요 Task**:
1. `run_katib_tuning`: Katib 실험 실행
2. `save_results`: 결과를 GCS에 저장
3. `send_notification`: 완료 알림 전송
4. `trigger_model_deployment`: 검증/배포 DAG 트리거

**출력**:
- `gs://{bucket}/vt-model/{site_id}/models/{yyyymm}/{yyyymm}_xgboost_{site_id}.json`
- `gs://{bucket}/vt-model/{site_id}/models/{yyyymm}/{yyyymm}_xgboost_{site_id}_model.pkl`

### 3. 모델 검증/배포 DAG (`dag_model_validation_deployment.py`)

**스케줄**: Katib DAG 완료 후 자동 트리거

**주요 Task**:
1. `check_new_model`: 새 모델 존재 확인
2. `validate_model`: 모델 성능 검증
3. `decide_deployment`: 배포 여부 결정
4. `promote_model`: 프로덕션 배포
5. `record_deployment`: 배포 이력 기록
6. `reload_kserve`: KServe 모델 리로드

**배포 조건**:
- RMSE 개선율 ≥ 2% (설정 가능)
- R² 저하율 ≤ 1% (설정 가능)

## 📁 디렉토리 구조

```
ML_pipeline_feedback_system/
├── README.md                          # 프로젝트 문서
├── SETUP.md                           # 상세 설정 가이드
│
├── config/                            # 설정 파일
│   ├── .env.example                   # 환경 변수 템플릿
│   ├── katib_config.yaml             # Katib 튜닝 설정
│   ├── data_download_config.yaml     # 데이터 다운로드 설정
│   └── deploy.yaml                   # 모델 배포 설정
│
├── pipeline/                          # Kubeflow 파이프라인
│   └── katib_pipeline.py             # Katib 튜닝 파이프라인
│
├── utils/                             # 유틸리티 모듈
│   ├── kubeflow_client.py            # Kubeflow API 클라이언트
│   ├── deployment.py                 # 모델 배포 유틸리티
│   └── common.py                     # 공통 유틸리티
│
├── k8s/                               # Kubernetes 매니페스트
│   ├── kserve_inferenceservice.yaml  # KServe 추론 서비스
│   └── gcs-data-sync-cronjob.yaml    # 데이터 동기화 CronJob
│
└── dag_*.py                           # Airflow DAG 파일
    ├── dag_katib_tuning.py           # Katib 튜닝 DAG
    ├── dag_model_validation_deployment.py  # 검증/배포 DAG
    └── dag_daily_data_download.py    # 데이터 다운로드 DAG
```

## 🔧 환경 변수

### 필수 환경 변수

| 변수명 | 설명 | 예시 |
|--------|------|------|
| `SITE_ID` | 사이트 고유 식별자 | `site1` |
| `HOST` | Kubeflow 호스트 URL | `https://kubeflow.example.com` |
| `USERNAME` | Kubeflow 사용자명 | `user@example.com` |
| `PASSWORD` | Kubeflow 비밀번호 | `********` |
| `NAMESPACE` | Kubernetes 네임스페이스 | `ml-production` |
| `GCS_BUCKET` | GCS 버킷명 | `my-ml-models` |
| `GCP_PROJECT` | GCP 프로젝트 ID | `my-gcp-project` |
| `KUBEFLOW_UI` | Kubeflow UI URL | `https://kubeflow.example.com` |

### 선택적 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `GCS_MODEL_BASE_PATH` | GCS 모델 저장 기본 경로 | `vt-model` |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCP 인증 파일 경로 | 설정 파일에서 읽음 |

## 🎯 주요 설정 파라미터

### Katib 튜닝 파라미터 (katib_config.yaml)

```yaml
defaults:
  kubeflow:
    max_trial_count: 12           # 최대 trial 수
    parallel_trial_count: 3       # 동시 실행 trial 수
    katib_timeout: 1200          # 타임아웃 (초)

    early_stopping:
      enabled: true
      algorithm: "medianstop"
      min_trials_required: 10
      start_step: 5
```

### XGBoost 하이퍼파라미터 탐색 범위

| 파라미터 | 범위 | 설명 |
|----------|------|------|
| `learning_rate` | 0.01 - 0.3 | 학습률 |
| `max_depth` | 3 - 10 | 트리 최대 깊이 |
| `n_estimators` | 100 - 3000 | 부스팅 라운드 수 |
| `subsample` | 0.5 - 0.8 | 샘플 비율 |
| `colsample_bytree` | 0.6 - 0.9 | 피처 비율 |
| `min_child_weight` | 1 - 10 | 최소 샘플 가중치 |
| `gamma` | 0.1 - 5.0 | 최소 손실 감소량 |

### 배포 검증 기준 (deploy.yaml)

```yaml
validation:
  min_improvement_rmse: 0.02    # RMSE 최소 개선율 (2%)
  max_degradation_r2: 0.01      # R² 최대 저하율 (1%)
```

## 🔍 모니터링

### Kubeflow UI
```bash
# Katib 실험 확인
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

# 브라우저에서 접속
open http://localhost:8080
```

### KServe 추론 서비스 상태
```bash
kubectl get inferenceservices -n your-namespace
```

### 배포 이력 확인
```python
from utils.deployment import get_deployment_history

# 최근 10개 배포 이력 조회
history = get_deployment_history(site_id='your_site_id', limit=10)
```

### GCS 저장 구조

```
gs://your-bucket/vt-model/{site_id}/
├── models/{yyyymm}/
│   ├── {yyyymm}_xgboost_{site_id}.json        # 베스트 하이퍼파라미터
│   ├── {yyyymm}_xgboost_{site_id}_model.pkl   # 베스트 모델
│   └── trials/
│       ├── trial-001_model.pkl
│       ├── trial-002_model.pkl
│       └── ...
│
└── model-deploy/
    ├── model.pkl                              # 프로덕션 모델
    ├── hyperparameters.json
    ├── metadata.json
    └── deployment_history.json                 # 배포 이력
```

## 🐛 트러블슈팅

### 1. Kubeflow 인증 실패
```bash
# 환경변수 확인
echo $HOST
echo $USERNAME

# .env 파일 확인
cat config/.env

# Kubeflow 클라이언트 테스트
python utils/kubeflow_client.py
```

### 2. GCS 접근 권한 오류
```bash
# Secret 확인
kubectl get secret gcp-credentials -n your-namespace

# Secret 재생성
kubectl delete secret gcp-credentials -n your-namespace
kubectl create secret generic gcp-credentials \
  --from-file=key.json=/path/to/credentials.json \
  -n your-namespace
```

### 3. Katib Trial 실패
```bash
# Trial Pod 로그 확인
kubectl logs -n your-namespace <trial-pod-name>

# Katib Experiment 상태 확인
kubectl get experiments -n your-namespace
kubectl describe experiment <experiment-name> -n your-namespace
```

### 4. KServe 배포 실패
```bash
# InferenceService 상태 확인
kubectl get inferenceservice -n your-namespace
kubectl describe inferenceservice xgboost-predictor -n your-namespace

# Pod 로그 확인
kubectl logs -n your-namespace -l serving.kserve.io/inferenceservice=xgboost-predictor
```

### 5. 설정 파일 오류
```bash
# YAML 문법 검증
python -c "import yaml; yaml.safe_load(open('config/katib_config.yaml'))"

# SITE_ID 일치 확인
echo $SITE_ID
grep -A 1 "sites:" config/katib_config.yaml
```

## 📚 참고 자료

- [Kubeflow Documentation](https://www.kubeflow.org/docs/)
- [Katib Guide](https://www.kubeflow.org/docs/components/katib/)
- [KServe Documentation](https://kserve.github.io/website/)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)

## 🤝 기여

이슈 및 PR은 언제나 환영합니다!

## 📝 라이선스

[License Type] - 자세한 내용은 LICENSE 파일을 참조하세요.

## 📞 문의

- **이메일**: ml-team@your-company.com
- **Slack**: #ml-pipeline-support
