"""
Count 계산 프로세서 모듈

SoCP(State of Cell Position) count를 계산하는 기능을 제공합니다.
"""

import pandas as pd
import os
import glob
from typing import Dict
from pytz import timezone
import sys

# sys.path에 현재 디렉토리의 부모 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SiteConfig, CHARGE_STATUS_LIST, SOC_RANGE_LIST
from core import DatabaseManager, FileManager


class CountProcessor:
    """
    SoCP count를 계산하는 프로세서
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

    def prepare_dataset(self, filename: str) -> None:
        """
        원본 데이터를 bank/rack별로 분할하여 저장

        Args:
            filename: 처리할 파일명 (날짜 형식: YYYY-MM-DD)
        """
        original_path = self.config.get_path('original')
        work_path = self.config.get_path('count_work')

        # 원본 데이터 로드
        if self.config.file_format == 'parquet':
            df = pd.read_parquet(os.path.join(original_path, f"{filename}.parquet"))
        else:
            df = pd.read_parquet(os.path.join(original_path, f"{filename}.parquet"))

        # UTC to Asia/Seoul
        df.set_index('TIMESTAMP', inplace=True)
        new_index = df.index.tz_convert(timezone("Asia/Seoul"))
        df.index = new_index

        # 작업 디렉토리 생성
        work_dir = os.path.join(work_path, filename)
        self.file_manager.ensure_directory(work_dir)

        # Bank/Rack별로 분할 저장
        bank_list = df["BANK_ID"].unique()

        for bank_id in bank_list:
            rack_list = df[df["BANK_ID"] == bank_id]["RACK_ID"].unique()

            for rack_id in rack_list:
                query_df = df.query(f"BANK_ID == {bank_id} and RACK_ID == {rack_id}")
                print(f"BANK_ID: {bank_id}, RACK_ID: {rack_id}")

                file_name = f"{self.config.site_name}{bank_id}_{rack_id}.csv"
                query_df.to_csv(os.path.join(work_dir, file_name))

    def calculate_count(self, filename: str, timestamp: str) -> None:
        """
        SoCP count 계산

        Args:
            filename: 파일명 (날짜 형식)
            timestamp: 타임스탬프 문자열
        """
        work_path = self.config.get_path('count_work')
        prep_path = self.config.get_path('count_prep')

        work_dir = os.path.join(work_path, filename)
        file_list = glob.glob(os.path.join(work_dir, "*"))

        merge_list = []

        for file_path in file_list:
            df = pd.read_csv(file_path)

            # 배터리 상태별로 분류
            battery_cols = self.config.get_battery_status_columns()

            # 상태 분류 (사이트마다 다를 수 있음)
            if "BATTERY_STATUS_FOR_CHARGE" in battery_cols:
                # baekma, gold 타입
                df_standby = df[df["BATTERY_STATUS_FOR_CHARGE"] == 1]
                df_charge = df[df["BATTERY_STATUS_FOR_CHARGE"] == 2]
                df_discharge = df[df["BATTERY_STATUS_FOR_CHARGE"] == 3]
            else:
                # panly 타입
                df_standby = df[df["BATTERY_STATUS_FOR_STANDBY"] == 1]
                df_charge = df[df["BATTERY_STATUS_FOR_CHARGE"] == 1]
                df_discharge = df[df["BATTERY_STATUS_FOR_DISCHARGE"] == 1]

            status_list = [df_charge, df_discharge, df_standby]

            charge_status = 0
            for cs_dataframe in status_list:
                for soc_range in SOC_RANGE_LIST:
                    result_dict = {
                        "PERIOD": self.config.period,
                        "TIMESTAMP": timestamp,
                        "BANK_ID": int(df["BANK_ID"].iloc[-1]),
                        "RACK_ID": int(df["RACK_ID"].iloc[-1]),
                        "CHARGE_STATUS": charge_status,
                        "SOC_RANGE": soc_range,
                    }

                    # 각 셀 위치별 count 계산
                    for cell_pos in range(1, self.config.cell_count + 1):
                        # 충전일 때는 MIN, 방전/대기일 때는 MAX
                        if charge_status == 0 or charge_status == 2:
                            condition = (
                                (cs_dataframe['RACK_SOC'] >= soc_range) &
                                (cs_dataframe['RACK_SOC'] < soc_range + 10) &
                                (cs_dataframe['RACK_MAX_CELL_VOLTAGE_POSITION'] == cell_pos)
                            )
                        elif charge_status == 1:
                            condition = (
                                (cs_dataframe['RACK_SOC'] >= soc_range) &
                                (cs_dataframe['RACK_SOC'] < soc_range + 10) &
                                (cs_dataframe['RACK_MIN_CELL_VOLTAGE_POSITION'] == cell_pos)
                            )

                        sub_df = cs_dataframe[condition]
                        count = len(sub_df)
                        result_dict[f"CELL_{cell_pos}"] = count

                    merge_list.append(result_dict)
                charge_status += 1

        # 결과 저장
        df_result = pd.DataFrame(merge_list)
        df_result = df_result.sort_values(
            by=['BANK_ID', 'RACK_ID', "CHARGE_STATUS", "SOC_RANGE"],
            ascending=True
        ).reset_index(drop=True)

        self.file_manager.ensure_directory(prep_path)
        df_result.to_csv(os.path.join(prep_path, f"{filename}.csv"))
        print(df_result)

    def save_to_database(self, filename: str) -> None:
        """
        계산 결과를 데이터베이스에 저장

        Args:
            filename: 파일명
        """
        prep_path = self.config.get_path('count_prep')
        table_name = self.config.get_table('count')

        df = pd.read_csv(os.path.join(prep_path, f"{filename}.csv"), index_col=0)
        self.db_manager.insert_dataframe(df, table_name)

    def cleanup_work_files(self) -> None:
        """작업 파일 정리"""
        work_path = self.config.get_path('count_work')
        self.file_manager.manage_old_files(work_path, max_files=7)

    def cleanup_prep_files(self) -> None:
        """전처리 파일 정리"""
        prep_path = self.config.get_path('count_prep')
        self.file_manager.manage_old_files(prep_path, max_files=7)
