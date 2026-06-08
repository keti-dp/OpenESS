"""
SoCP DAGs - 중앙 집중식 DAG 생성

이 모듈은 여러 개별 DAG 파일을 단일하고 유지 관리 가능한
Factory 기반 접근 방식으로 통합합니다. 모든 DAG는 Factory 패턴을 사용하여 동적으로 생성됩니다.

사이트: baekma, gold, panly
SoCP DAG 유형: GetDataset, Count, Info, 1st Differencing, Moving Average

DAG 구조:
1. SoCP_master (1개 마스터 DAG)
   - 스케줄에 따라 실행
   - 모든 사이트의 GetDataset DAG를 병렬로 트리거

2. {site}_SoCP_GetDataset (3개 DAG)
   - 마스터 DAG에 의해 트리거됨
   - 데이터를 가져와 저장
   - Count 및 Info DAG를 트리거

3. {site}_SoCP_count (3개 DAG)
   - GetDataset DAG에 의해 트리거됨
   - SoCP count 계산
   - Differencing 및 Moving Average DAG를 트리거

4. {site}_SoCP_info (3개 DAG)
   - GetDataset DAG에 의해 트리거됨
   - SoCP info 계산 (1일, 7일, 30일)

5. {site}_SoCP_1st_differencing (3개 DAG)
   - Count DAG에 의해 트리거됨
   - 1차 차분 계산

6. {site}_SoCP_moving_average (3개 DAG)
   - Count DAG에 의해 트리거됨
   - 이동 평균 계산 (5일, 10일, 15일, 30일)

총: 16개 DAG (마스터 1개 + 사이트별 5개 × 3사이트)

Dependencies 흐름:
SoCP_master → {site}_SoCP_GetDataset → {site}_SoCP_count → {site}_SoCP_1st_differencing
                                     ↓                    ↓
                              {site}_SoCP_info    {site}_SoCP_moving_average
"""

import sys
from pathlib import Path

# DAG 파일의 디렉토리를 Python 경로에 추가
DAG_DIR = Path(__file__).parent.resolve()
if str(DAG_DIR) not in sys.path:
    sys.path.insert(0, str(DAG_DIR))

from dag_factory import generate_all_dags

# 모든 DAG를 생성하고 전역 네임스페이스에 주입
# 이를 통해 Airflow의 DAG parser가 인식할 수 있게 함
all_dags = generate_all_dags()
globals().update(all_dags)

# 디버깅용 요약 출력
if __name__ == "__main__":
    print(f"생성된 DAG 개수: {len(all_dags)}")
    for dag_id in sorted(all_dags.keys()):
        print(f"  - {dag_id}")
