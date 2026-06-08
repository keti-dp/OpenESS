"""VIExTD (전압 증감 극값 시간 차이) 계산기"""

from typing import Dict, Any
from datetime import timedelta
import pandas as pd

from calculators.base_calculator import BaseHealthIndicatorCalculator


class VIExTDCalculator(BaseHealthIndicatorCalculator):
    """VIExTD 건강 지표 계산기"""

    def get_table_name(self) -> str:
        """데이터베이스 테이블 이름을 가져옵니다."""
        return "health_indicator_viextd"

    def get_insert_query(self) -> str:
        """SQL insert 쿼리 템플릿을 가져옵니다."""
        return """
            INSERT INTO health_indicator_viextd (
                "TIMESTAMP",
                "OPERATING_SITE",
                "BANK_ID",
                "RACK_ID",
                "S_VIECTD",
                "E_VIECTD",
                "VIECTD",
                "S_VIEDTD",
                "E_VIEDTD",
                "VIEDTD"
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

    def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[int, Dict[int, Any]]:
        """각 bank와 rack에 대해 VIExTD를 계산합니다.

        VIExTD는 시간에 따른 전압 변화를 측정합니다:
        - VIECTD: 충전 시 전압 증가 극값 시간 차이
        - VIEDTD: 방전 시 전압 감소 극값 시간 차이

        Args:
            df: 배터리 데이터가 포함된 입력 DataFrame
            **kwargs: 추가 파라미터

        Returns:
            중첩된 dict: {bank_id: {rack_id: {metrics}}}
        """
        result_value = {}
        soc = 60
        time = 1000  # 시간 간격(초)
        bank_list = df["BANK_ID"].unique()

        for bank_id in bank_list:
            result_value[int(bank_id)] = {}
            filtered_df = df.query(f"BANK_ID == {bank_id}")
            rack_list = filtered_df["RACK_ID"].unique()

            for rack_id in rack_list:
                try:
                    result_value[int(bank_id)][int(rack_id)] = {}

                    # 사이트의 배터리 상태 필드를 기반으로 충전 및 방전 데이터 쿼리
                    if 'BATTERY_STATUS_FOR_STANDBY' in filtered_df.columns:
                        # Gold/Panly 스타일
                        viectd_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_CHARGE == 1"
                        ).reset_index(drop=True)
                        viedtd_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_DISCHARGE == 1"
                        ).reset_index(drop=True)
                    elif 'BATTERY_STATUS_OF_STAND_BY' in filtered_df.columns:
                        # Seokhwan 스타일
                        viectd_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_CHARGE == 1"
                        ).reset_index(drop=True)
                        viedtd_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_DISCHARGE == 1"
                        ).reset_index(drop=True)
                    elif 'BATTERY_STATUS_FOR_RUN' in filtered_df.columns:
                        # Baekma/Seongdeok 스타일
                        viectd_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_RUN == 1 and BATTERY_STATUS_FOR_CHARGE == 2"
                        ).reset_index(drop=True)
                        viedtd_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_RUN == 1 and BATTERY_STATUS_FOR_CHARGE == 3"
                        ).reset_index(drop=True)
                    else:
                        raise KeyError(
                            f"Required status columns not found. "
                            f"Available columns: {filtered_df.columns.tolist()}"
                        )

                    # TIMESTAMP를 index로 설정하고 중복 제거
                    viectd_df = viectd_df.set_index('TIMESTAMP')
                    viedtd_df = viedtd_df.set_index('TIMESTAMP')
                    viectd_df = viectd_df.loc[~viectd_df.index.duplicated(keep='first')]
                    viedtd_df = viedtd_df.loc[~viedtd_df.index.duplicated(keep='first')]

                    # 충전 시 SOC 60% 데이터 또는 가장 가까운 데이터 찾기
                    if viectd_df.query(f"RACK_SOC == {soc}").empty:
                        closest_row = (viectd_df['RACK_SOC'] - soc).abs().idxmin()
                        viectd_soc = viectd_df.loc[[closest_row]]
                    else:
                        viectd_soc = viectd_df.query(f"RACK_SOC == {soc}")

                    # 방전 시 SOC 60% 데이터 또는 가장 가까운 데이터 찾기
                    if viedtd_df.query(f"RACK_SOC == {soc}").empty:
                        closest_row = (viedtd_df['RACK_SOC'] - soc).abs().idxmin()
                        viedtd_soc = viedtd_df.loc[[closest_row]]
                    else:
                        viedtd_soc = viedtd_df.query(f"RACK_SOC == {soc}")

                    # 초기 전압 가져오기
                    init_viectd_v = viectd_soc.loc[viectd_soc.index[0], "RACK_VOLTAGE"]
                    init_viedtd_v = viedtd_soc.loc[viedtd_soc.index[0], "RACK_VOLTAGE"]

                    # n초 후 시간 계산
                    viectd_e_time = viectd_soc.index[0] + timedelta(seconds=time)
                    viedtd_e_time = viedtd_soc.index[0] + timedelta(seconds=time)

                    # 가장 가까운 타임스탬프 찾기
                    viectd_nearest_index = viectd_df.index.get_indexer([viectd_e_time], method='nearest')
                    viedtd_nearest_index = viedtd_df.index.get_indexer([viedtd_e_time], method='nearest')

                    # 가장 가까운 타임스탬프에서 데이터 가져오기
                    viectd_nearest_data = viectd_df.iloc[viectd_nearest_index]
                    viedtd_nearest_data = viedtd_df.iloc[viedtd_nearest_index]

                    # 전압 차이 계산
                    viectd = round(abs(viectd_nearest_data["RACK_VOLTAGE"].values[0] - init_viectd_v), 3)
                    viedtd = round(abs(viedtd_nearest_data["RACK_VOLTAGE"].values[0] - init_viedtd_v), 3)

                    print(f"BANK_ID: {bank_id}, RACK_ID: {rack_id}, VIECTD: {viectd}, VIEDTD: {viedtd}")

                    result_value[int(bank_id)][int(rack_id)] = {
                        "VIECTD": viectd,
                        "VIEDTD": viedtd,
                        "S_VIECTD": pd.Timestamp(viectd_soc.index[0]).strftime('%Y-%m-%d %H:%M:%S'),
                        "E_VIECTD": pd.Timestamp(viectd_nearest_data["RACK_VOLTAGE"].index.values[0]).strftime('%Y-%m-%d %H:%M:%S'),
                        "S_VIEDTD": pd.Timestamp(viedtd_soc.index[0]).strftime('%Y-%m-%d %H:%M:%S'),
                        "E_VIEDTD": pd.Timestamp(viedtd_nearest_data["RACK_VOLTAGE"].index.values[0]).strftime('%Y-%m-%d %H:%M:%S'),
                    }

                except ValueError:
                    print(f"오류: {bank_id}_{rack_id} VIExTD 계산에 필요한 데이터 없음")
                    continue

        return result_value
