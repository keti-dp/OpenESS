"""
Moving Average 계산 프로세서 모듈

이동 평균(Moving Average)을 계산하는 기능을 제공합니다.
"""

import pandas as pd
import os
from typing import List
import sys

# sys.path에 현재 디렉토리의 부모 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SiteConfig, CHARGE_STATUS_LIST, SOC_RANGE_LIST
from core import DatabaseManager, FileManager


class MovingAvgProcessor:
    """
    Moving Average를 계산하는 프로세서
    """

    # 이동 평균 윈도우 크기 목록
    MA_WINDOWS = [5, 10, 15, 30]

    def __init__(self, site_config: SiteConfig):
        """
        초기화

        Args:
            site_config: 사이트 설정 객체
        """
        self.config = site_config
        self.db_manager = DatabaseManager(site_config.stats_conn_id)
        self.file_manager = FileManager()

    def fetch_source_data(
        self,
        start_time: str,
        end_time: str,
        filename: str,
        period: int
    ) -> None:
        """
        소스 데이터를 데이터베이스에서 가져와 저장

        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            filename: 저장할 파일명
            period: 기간 (5, 10, 15, 30)
        """
        save_path = self.config.get_path('moving_avg')

        # 디렉토리 생성
        file_dir = self.file_manager.create_filename_directory(save_path, filename)

        # 데이터 조회
        table_name = self.config.get_table('count')
        query = f"""SELECT * FROM {table_name} WHERE ("TIMESTAMP" BETWEEN '{start_time}' AND '{end_time}')"""

        result, column_names = self.db_manager.execute_query_with_columns(query)

        # DataFrame 생성 및 저장
        df = pd.DataFrame(result, columns=column_names)
        print(df)
        df.to_csv(os.path.join(file_dir, f"{filename}_{period}.csv"), index=False)

    def calculate_moving_average(self, filename: str, ma_window: int) -> None:
        """
        Moving Average 계산

        Args:
            filename: 파일명
            ma_window: 이동 평균 윈도우 크기
        """
        save_path = self.config.get_path('moving_avg')
        file_path = os.path.join(save_path, filename, f"{filename}_{ma_window}.csv")

        result_df = pd.DataFrame()

        df = pd.read_csv(file_path, index_col="TIMESTAMP", parse_dates=["TIMESTAMP"])
        df = df.sort_index()

        # CELL로 시작하는 컬럼 찾기
        cell_columns = [col for col in df.columns if 'CELL' in col]

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

                        # Rolling window를 사용한 이동 평균 계산
                        df_temp = calc_df[cell_columns].rolling(window=ma_window).mean()
                        df_temp.reset_index(inplace=True)

                        # 메타 컬럼 추가
                        df_temp[['PERIOD', 'BANK_ID', 'RACK_ID', 'CHARGE_STATUS', 'SOC_RANGE']] = \
                            calc_df[['PERIOD', 'BANK_ID', 'RACK_ID', 'CHARGE_STATUS', 'SOC_RANGE']].values

                        # MA 윈도우 크기 컬럼 추가
                        df_temp['MA'] = ma_window

                        # 해당 날짜의 데이터만 필터링
                        target_date = f"{filename[:4]}-{filename[4:6]}-{filename[6:8]}"
                        df_temp = df_temp[df_temp["TIMESTAMP"] == target_date]

                        result_df = pd.concat([result_df, df_temp])

        result_df = result_df.reset_index(drop=True)
        print(result_df)

        result_path = os.path.join(save_path, filename, f"result_{ma_window}.csv")
        result_df.to_csv(result_path, index=False)

    def save_to_database(self, filename: str) -> None:
        """
        모든 MA 결과를 데이터베이스에 저장

        Args:
            filename: 파일명
        """
        save_path = self.config.get_path('moving_avg')
        table_name = self.config.get_table('moving_avg')

        file_dir = os.path.join(save_path, filename)
        file_list = os.listdir(file_dir)
        result_files = [f for f in file_list if f.startswith('result_')]

        # 모든 결과 파일 합치기
        tmp_df = pd.DataFrame()
        for f_name in result_files:
            df = pd.read_csv(os.path.join(file_dir, f_name))
            tmp_df = pd.concat([tmp_df, df])

        tmp_df = tmp_df.reset_index(drop=True)
        print(tmp_df)

        self.db_manager.insert_dataframe(tmp_df, table_name)

    def cleanup_files(self) -> None:
        """파일 정리"""
        save_path = self.config.get_path('moving_avg')
        self.file_manager.manage_old_files(save_path, max_files=7)
