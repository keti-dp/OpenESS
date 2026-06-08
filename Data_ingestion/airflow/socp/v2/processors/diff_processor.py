"""
1st Differencing 계산 프로세서 모듈

시계열 데이터의 1차 차분을 계산하는 기능을 제공합니다.
"""

import pandas as pd
import os
from typing import Dict
import sys

# sys.path에 현재 디렉토리의 부모 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SiteConfig, CHARGE_STATUS_LIST, SOC_RANGE_LIST
from core import DatabaseManager, FileManager


class DiffProcessor:
    """
    1st differencing을 계산하는 프로세서
    """

    def __init__(self, site_config: SiteConfig):
        """
        초기화

        Args:
            site_config: 사이트 설정 객체
        """
        self.config = site_config
        self.db_manager = DatabaseManager(site_config.stats_conn_id)
        self.file_manager = FileManager()

    def fetch_source_data(self, start_time: str, end_time: str, filename: str) -> None:
        """
        소스 데이터를 데이터베이스에서 가져와 저장

        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            filename: 저장할 파일명
        """
        save_path = self.config.get_path('count_diff')

        # 디렉토리 생성
        file_dir = self.file_manager.create_filename_directory(save_path, filename)

        # 데이터 조회
        table_name = self.config.get_table('count')
        query = f"""SELECT * FROM {table_name} WHERE ("TIMESTAMP" BETWEEN '{start_time}' AND '{end_time}')"""

        result, column_names = self.db_manager.execute_query_with_columns(query)

        # DataFrame 생성 및 저장
        df = pd.DataFrame(result, columns=column_names)
        print(df)
        df.to_csv(os.path.join(file_dir, f"{filename}.csv"), index=False)

    def calculate_differencing(self, filename: str) -> None:
        """
        1차 차분 계산

        Args:
            filename: 파일명
        """
        save_path = self.config.get_path('count_diff')
        file_path = os.path.join(save_path, filename, f"{filename}.csv")

        result_df = pd.DataFrame()

        df = pd.read_csv(file_path, index_col="TIMESTAMP", parse_dates=["TIMESTAMP"])
        df = df.sort_index()

        bank_list = df["BANK_ID"].unique()
        for bank_id in bank_list:
            rack_list = df[df["BANK_ID"] == bank_id]["RACK_ID"].unique()

            for rack_id in rack_list:
                filtered_df = df.query(f"BANK_ID == {bank_id} and RACK_ID == {rack_id}")

                for charge_status in CHARGE_STATUS_LIST:
                    for soc_range in SOC_RANGE_LIST:
                        calc_df = filtered_df.query(
                            f"CHARGE_STATUS == {charge_status} and SOC_RANGE == {soc_range}"
                        )

                        # 차분 결과를 저장할 dict
                        diff_data = {}

                        # 각 셀에 대해 차분 계산
                        for i in range(1, self.config.cell_count + 1):
                            cell_col = f'CELL_{i}'
                            calc_df_copy = calc_df.copy()

                            # shift 함수를 사용하여 이전 날짜의 값을 얻음
                            calc_df_copy[f'PREV_{cell_col}'] = calc_df_copy[cell_col].shift(1)

                            # diff 함수를 사용하여 이전날과 다음날의 차이를 계산
                            diff_data[f'DIFF_{cell_col}'] = (
                                calc_df_copy[cell_col] - calc_df_copy[f'PREV_{cell_col}']
                            )

                        # 기존 데이터프레임에서 필요한 열들을 선택
                        preserve_cols = ['PERIOD', 'BANK_ID', 'RACK_ID', 'CHARGE_STATUS', 'SOC_RANGE']
                        preserve_df = calc_df_copy[preserve_cols].copy()

                        # 차이를 계산한 데이터와 원래의 필요한 열들을 합침
                        diff_df = pd.concat([preserve_df, pd.DataFrame(diff_data)], axis=1)
                        diff_df.index.name = 'TIMESTAMP'

                        # 인덱스를 컬럼으로 변환
                        diff_df.reset_index(inplace=True)

                        # 해당 날짜의 데이터만 필터링
                        target_date = f"{filename[:4]}-{filename[4:6]}-{filename[6:8]}"
                        diff_df = diff_df[diff_df["TIMESTAMP"] == target_date]

                        result_df = pd.concat([result_df, diff_df])

        print(result_df)
        result_path = os.path.join(save_path, filename, "result.csv")
        result_df.to_csv(result_path, index=False)

    def save_to_database(self, filename: str) -> None:
        """
        계산 결과를 데이터베이스에 저장

        Args:
            filename: 파일명
        """
        save_path = self.config.get_path('count_diff')
        table_name = self.config.get_table('count_diff')

        result_path = os.path.join(save_path, filename, "result.csv")
        df = pd.read_csv(result_path)

        self.db_manager.insert_dataframe(df, table_name)

    def cleanup_files(self) -> None:
        """파일 정리"""
        save_path = self.config.get_path('count_diff')
        self.file_manager.manage_old_files(save_path, max_files=7)
