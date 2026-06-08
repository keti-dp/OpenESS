"""
건강 지표 DAGs - 중앙 집중식 DAG 생성

이 모듈은 18개의 개별 DAG 파일을 단일하고 유지 관리 가능한
Factory 기반 접근 방식으로 대체합니다. 모든 DAG는 Factory 패턴을 사용하여 동적으로 생성됩니다.

사이트: baekma, gold, panly
건강 지표: DCIR, MVF, PE, TIExVD, VIExTD

DAG 구조:
1. Health_Indicator_master (1개 마스터 DAG)
   - 매일 새벽 5시에 스케줄 실행
   - 모든 사이트의 dataset DAG를 병렬로 트리거

2. {site}_HI_get_dataset (3개 DAG)
   - 마스터 DAG에 의해 트리거됨
   - 각 건강 지표 계산 DAG를 트리거

3. {site}_HI_calc_{indicator} (15개 DAG)
   - dataset DAG에 의해 트리거됨

총: 19개 DAG (마스터 1개 + 데이터셋 3개 + 계산 15개)

Dependencies 흐름:
Health_Indicator_master → {site}_HI_get_dataset → {site}_HI_calc_{indicator}
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
