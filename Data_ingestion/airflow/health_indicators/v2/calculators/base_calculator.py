"""건강 지표 계산기를 위한 기본 클래스"""

import sys
from pathlib import Path

# DAG 파일의 디렉토리를 Python 경로에 추가
DAG_DIR = Path(__file__).parent.parent.resolve()
if str(DAG_DIR) not in sys.path:
    sys.path.insert(0, str(DAG_DIR))

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

from config.site_config import SiteConfig


class BaseHealthIndicatorCalculator(ABC):
    """건강 지표 계산기를 위한 추상 기본 클래스"""

    def __init__(self, site_config: SiteConfig):
        """사이트 설정으로 계산기를 초기화합니다.

        Args:
            site_config: 사이트 설정 객체
        """
        self.site_config = site_config

    def load_dataset(self, data_save_path: str, filename: str) -> pd.DataFrame:
        """parquet 데이터셋을 로드합니다.

        Args:
            data_save_path: 데이터셋 디렉토리 경로
            filename: 확장자를 제외한 파일명

        Returns:
            데이터셋이 포함된 DataFrame
        """
        filepath = f"{data_save_path}{filename}.parquet"
        return pd.read_parquet(filepath)

    @abstractmethod
    def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[int, Dict[int, Any]]:
        """건강 지표를 계산합니다.

        Args:
            df: 입력 DataFrame
            **kwargs: 추가 파라미터

        Returns:
            중첩된 dict: {bank_id: {rack_id: {metric: value}}}
        """
        pass

    @abstractmethod
    def get_table_name(self) -> str:
        """이 지표에 대한 데이터베이스 테이블 이름을 가져옵니다.

        Returns:
            테이블 이름 문자열
        """
        pass

    @abstractmethod
    def get_insert_query(self) -> str:
        """SQL insert 쿼리 템플릿을 가져옵니다.

        Returns:
            플레이스홀더가 포함된 SQL 쿼리 문자열
        """
        pass

    def execute_calculation(self, **context) -> None:
        """계산을 실행하고 결과를 XCom에 push합니다.

        Args:
            **context: Airflow context
        """
        data_save_path = context["task_instance"].xcom_pull(task_ids="initialize_globals", key="DATA_SAVE_PATH")
        query_time = context["task_instance"].xcom_pull(task_ids="calc_time_range", key="query_time")
        filename = str(query_time["start_time"])[:10]

        df = self.load_dataset(data_save_path, filename)
        result = self.calculate(df)

        context["task_instance"].xcom_push(key="result", value=result)
        print(f"계산 완료. 결과: {result}")
