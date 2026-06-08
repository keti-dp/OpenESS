"""
Info 계산 프로세서 모듈

여러 기간(1일, 7일, 30일)에 대한 SoCP 정보를 계산하는 기능을 제공합니다.
"""

import pandas as pd
import os
import glob
import copy
from typing import Dict, List
from pytz import timezone
import sys

# sys.path에 현재 디렉토리의 부모 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SiteConfig, CHARGE_STATUS_LIST, SOC_RANGE_LIST
from core import DatabaseManager, FileManager


class InfoProcessor:
    """
    SoCP 정보를 계산하는 프로세서
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

    def load_and_split_dataset(
        self,
        start_date: str,
        end_date: str,
        period_name: str
    ) -> None:
        """
        기간별 데이터를 로드하고 bank/rack별로 분할 저장

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            period_name: 기간 이름 (예: 'past_1days', 'past_7days')
        """
        original_path = self.config.get_path('original')
        period_path = self.config.get_path('info_period')

        filename = end_date[:10]

        # 날짜 범위 생성
        date_range = pd.date_range(start=start_date, end=end_date).strftime('%Y-%m-%d').tolist()

        # 여러 날짜의 데이터를 하나로 합치기
        list_of_df = []
        for date in date_range:
            try:
                if self.config.file_format == 'parquet':
                    df = pd.read_parquet(os.path.join(original_path, f"{date}.parquet"))
                else:
                    df = pd.read_parquet(os.path.join(original_path, f"{date}.parquet"))
                list_of_df.append(df)
            except FileNotFoundError:
                print(f"{date} 데이터 없음!")
                continue

        if not list_of_df:
            print(f"{period_name}: 로드된 데이터가 없습니다.")
            return

        df_accum = pd.concat(list_of_df).reset_index(drop=True)
        print(df_accum)

        # 저장 디렉토리 생성
        save_dir = os.path.join(period_path, filename)
        self.file_manager.ensure_directory(save_dir)

        # Bank/Rack별로 분할 저장
        bank_list = df_accum["BANK_ID"].unique()
        for bank in bank_list:
            query_bank_df = df_accum.query(f"BANK_ID == {bank}")
            rack_list = query_bank_df["RACK_ID"].unique()

            for rack in rack_list:
                query_rack_df = query_bank_df.query(f"RACK_ID == {rack}").reset_index(drop=True)
                file_name = f"{period_name}_{bank}_{rack}.parquet"
                query_rack_df.to_parquet(os.path.join(save_dir, file_name))

    def calculate_info(
        self,
        filename: str,
        period_name: str,
        period_value: int,
        timestamp: str
    ) -> pd.DataFrame:
        """
        SoCP 정보 계산

        Args:
            filename: 파일명
            period_name: 기간 이름
            period_value: 기간 값 (1, 7, 30)
            timestamp: 타임스탬프

        Returns:
            계산 결과 DataFrame
        """
        period_path = self.config.get_path('info_period')
        period_dir = os.path.join(period_path, filename)

        file_pattern = os.path.join(period_dir, f"{period_name}_*.parquet")
        file_list = glob.glob(file_pattern)

        tmp_list = []

        for file_path in file_list:
            df = pd.read_parquet(file_path)

            # 배터리 상태별로 분류
            battery_cols = self.config.get_battery_status_columns()

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
                        "OPERATING_SITE": self.config.site_name,
                        "PERIOD": period_value,
                        "TIMESTAMP": timestamp,
                        "BANK_ID": int(df["BANK_ID"].iloc[-1]) if len(df) > 0 else 0,
                        "RACK_ID": int(df["RACK_ID"].iloc[-1]) if len(df) > 0 else 0,
                        "CHARGE_STATUS": charge_status,
                        "SOC_RANGE": soc_range,
                    }

                    # 각 셀 위치별 count 계산
                    for cell_pos in range(1, self.config.cell_count + 1):
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

                    tmp_list.append(result_dict)
                charge_status += 1

        return pd.DataFrame(tmp_list)

    def merge_and_save_results(
        self,
        filename: str,
        results: Dict[str, pd.DataFrame]
    ) -> None:
        """
        여러 기간의 결과를 병합하여 저장

        Args:
            filename: 파일명
            results: 기간별 결과 딕셔너리 {period_name: DataFrame}
        """
        prep_path = self.config.get_path('info_prep')
        self.file_manager.ensure_directory(prep_path)

        # 모든 결과 합치기
        all_results = []
        for period_name, df in results.items():
            all_results.append(df)

        if all_results:
            merged_df = pd.concat(all_results, ignore_index=True)
            merged_df = merged_df.sort_values(
                by=['PERIOD', 'BANK_ID', 'RACK_ID', 'CHARGE_STATUS', 'SOC_RANGE'],
                ascending=True
            ).reset_index(drop=True)

            merged_df.to_csv(os.path.join(prep_path, f"{filename}.csv"))
            print(merged_df)

    def save_to_database(self, filename: str) -> None:
        """
        계산 결과를 데이터베이스에 저장

        Args:
            filename: 파일명
        """
        prep_path = self.config.get_path('info_prep')
        table_name = self.config.get_table('info')

        df = pd.read_csv(os.path.join(prep_path, f"{filename}.csv"), index_col=0)
        self.db_manager.insert_dataframe(df, table_name)

    def cleanup_period_files(self) -> None:
        """기간별 파일 정리"""
        period_path = self.config.get_path('info_period')
        self.file_manager.manage_old_files(period_path, max_files=7)

    def cleanup_prep_files(self) -> None:
        """전처리 파일 정리"""
        prep_path = self.config.get_path('info_prep')
        self.file_manager.manage_old_files(prep_path, max_files=7)
