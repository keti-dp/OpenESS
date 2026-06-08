"""
데이터셋 처리 모듈

원본 데이터베이스에서 데이터를 가져와 파일로 저장하는 기능을 제공합니다.
"""

import pandas as pd
from typing import Optional
import sys
import os

# sys.path에 현재 디렉토리의 부모 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SiteConfig
from core import DatabaseManager, FileManager


class DatasetProcessor:
    """
    원본 데이터셋을 가져와 저장하는 프로세서
    """

    def __init__(self, site_config: SiteConfig):
        """
        초기화

        Args:
            site_config: 사이트 설정 객체
        """
        self.config = site_config
        self.db_manager = DatabaseManager(site_config.db_conn_id)
        self.file_manager = FileManager()

    def fetch_and_save_dataset(
        self,
        start_time: str,
        end_time: str
    ) -> Optional[str]:
        """
        데이터베이스에서 데이터를 조회하여 파일로 저장

        Args:
            start_time: 조회 시작 시간
            end_time: 조회 종료 시간

        Returns:
            저장된 파일명 (데이터가 없으면 None)
        """
        # 쿼리 생성
        battery_status_cols = self.config.get_battery_status_columns()
        query = self.db_manager.build_rack_query(
            start_time, end_time, battery_status_cols
        )

        # 데이터 조회
        df = self.db_manager.execute_query_to_dataframe(query)

        if df.empty:
            print(f"조회된 데이터가 없습니다: {start_time} ~ {end_time}")
            return None

        # 데이터 정렬
        df = df.sort_values(
            by=['TIMESTAMP', 'BANK_ID', 'RACK_ID'],
            ascending=True
        ).reset_index(drop=True)

        # 파일명 생성
        filename = start_time[:10]

        try:
            # 타임존 변환
            df["TIMESTAMP"] = df["TIMESTAMP"].dt.tz_convert('Asia/Seoul')

            # 파일 저장
            output_path = self.config.get_path('original')
            self.file_manager.ensure_directory(output_path)

            if self.config.file_format == 'parquet':
                df.to_parquet(os.path.join(output_path, f"{filename}.parquet"))
            elif self.config.file_format == 'feather':
                df.to_feather(os.path.join(output_path, f"{filename}.feather"))
            else:
                raise ValueError(f"지원하지 않는 파일 포맷: {self.config.file_format}")

            print(f"데이터셋 저장 완료: {filename}")
            return filename

        except AttributeError as e:
            print(f"타임존 변환 오류 ({filename}): {e}")
            return None

    def cleanup_old_files(self, max_files: int = 30) -> None:
        """
        오래된 원본 파일 삭제

        Args:
            max_files: 유지할 최대 파일 개수 (기본값: 30)
        """
        output_path = self.config.get_path('original')
        self.file_manager.manage_old_files(output_path, max_files)
