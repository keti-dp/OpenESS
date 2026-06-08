"""
상수 정의 모듈

프로젝트 전체에서 사용되는 공통 상수를 정의합니다.
"""

from typing import List

# 충전 상태 리스트
# 0: 충전, 1: 방전, 2: 대기
CHARGE_STATUS_LIST: List[int] = [0, 1, 2]

# SOC 범위 리스트 (0~100을 10 단위로 분할)
SOC_RANGE_LIST: List[int] = list(range(0, 100, 10))

# 기본 타임존
DEFAULT_TIMEZONE: str = "Asia/Seoul"

# 파일 보관 일수
FILE_RETENTION_DAYS: int = 7
FILE_RETENTION_DAYS_ORIGINAL: int = 30
