"""PE (부분 에너지) 계산기"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from calculators.base_calculator import BaseHealthIndicatorCalculator


class PECalculator(BaseHealthIndicatorCalculator):
    """Partial Energy 건강 지표 계산기"""

    def get_table_name(self) -> str:
        """데이터베이스 테이블 이름을 가져옵니다."""
        return "health_indicator_pe"

    def get_insert_query(self) -> str:
        """SQL insert 쿼리 템플릿을 가져옵니다."""
        return """
            INSERT INTO health_indicator_pe (
                "TIMESTAMP",
                "OPERATING_SITE",
                "BANK_ID",
                "RACK_ID",
                "S_TIME",
                "E_TIME",
                "MIN_SOC",
                "MAX_SOC",
                "MIN_VOLTAGE",
                "MAX_VOLTAGE",
                "MIN_CURRENT",
                "MAX_CURRENT",
                "PE"
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

    def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[int, Dict[int, Any]]:
        """각 bank와 rack에 대해 Partial Energy를 계산합니다.

        Args:
            df: 배터리 데이터가 포함된 입력 DataFrame
            **kwargs: 추가 파라미터

        Returns:
            중첩된 dict: {bank_id: {rack_id: {metrics}}}
        """
        result_value = {}
        soc = 60
        bank_list = df["BANK_ID"].unique()

        for bank_id in bank_list:
            result_value[int(bank_id)] = {}
            bank_df = df.query(f"BANK_ID == {bank_id}")

            for rack_id in bank_df["RACK_ID"].unique():
                result_value[int(bank_id)][int(rack_id)] = {}

                # 사이트의 배터리 상태 필드를 기반으로 충전 데이터 쿼리
                if 'BATTERY_STATUS_FOR_STANDBY' in bank_df.columns:
                    # Gold/Panly 스타일
                    rack_pe_df = bank_df.query(
                        f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_CHARGE == 1"
                    ).reset_index(drop=True)
                elif 'BATTERY_STATUS_OF_STAND_BY' in bank_df.columns:
                    # Seokhwan 스타일
                    rack_pe_df = bank_df.query(
                        f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_CHARGE == 1"
                    ).reset_index(drop=True)
                elif 'BATTERY_STATUS_FOR_RUN' in bank_df.columns:
                    # Baekma/Seongdeok 스타일
                    rack_pe_df = bank_df.query(
                        f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_RUN == 1 and BATTERY_STATUS_FOR_CHARGE == 2"
                    ).reset_index(drop=True)
                else:
                    raise KeyError(
                        f"Required status columns not found. "
                        f"Available columns: {bank_df.columns.tolist()}"
                    )

                # TIMESTAMP를 index로 설정하고 중복 제거
                rack_pe_df = rack_pe_df.set_index('TIMESTAMP')
                rack_pe_df = rack_pe_df.loc[~rack_pe_df.index.duplicated(keep='first')]

                try:
                    # SOC 60% 데이터 또는 가장 가까운 데이터 찾기
                    if rack_pe_df.query(f"RACK_SOC == {soc}").empty:
                        closest_row = (rack_pe_df['RACK_SOC'] - soc).abs().idxmin()
                        pe_soc = rack_pe_df.loc[[closest_row]]
                    else:
                        pe_soc = rack_pe_df.query(f"RACK_SOC == {soc}")

                except ValueError:
                    print(f"오류: {bank_id}_{rack_id} PE 계산에 필요한 데이터 없음")
                    continue

                saved_soc = pe_soc["RACK_SOC"].unique()[0]

                # SOC 차이가 너무 크면 건너뛰기
                if abs(saved_soc - soc) > self.site_config.diff_soc:
                    print(f"BANK_ID: {bank_id}, RACK_ID: {rack_id} SOC: {saved_soc} (건너뜀)")
                    continue

                # SOC 60에서의 최소 전압 가져오기
                vmin_at_soc_60 = pe_soc["RACK_VOLTAGE"].min()

                # SOC 60부터 전압 + 10V까지 데이터 쿼리
                v_plus_10 = vmin_at_soc_60 + 10
                pe_df = rack_pe_df.query(
                    f"RACK_SOC >= {saved_soc} & RACK_VOLTAGE >= {vmin_at_soc_60} & RACK_VOLTAGE <= {v_plus_10}"
                )

                # 누락된 타임스탬프를 forward fill로 채우기
                min_time = pe_df.index.min()
                max_time = pe_df.index.max()
                time_range = pd.date_range(start=min_time, end=max_time, freq='S')
                pe_df = pe_df.reindex(time_range).fillna(method='ffill')

                # 부분 에너지 계산
                voltages = pe_df["RACK_VOLTAGE"].to_numpy()
                currents = pe_df["RACK_CURRENT"].to_numpy()
                delta_t = 1  # 1초
                partial_energy = round(np.sum(voltages * currents * delta_t) / 3600, 4)

                s_time = pe_df.index.min().strftime('%Y-%m-%d %H:%M:%S')
                e_time = pe_df.index.max().strftime('%Y-%m-%d %H:%M:%S')
                v_min = pe_df["RACK_VOLTAGE"].min()
                i_min = pe_df["RACK_CURRENT"].min()
                v_max = pe_df["RACK_VOLTAGE"].max()
                i_max = pe_df["RACK_CURRENT"].max()
                soc_min = pe_df["RACK_SOC"].min()
                soc_max = pe_df["RACK_SOC"].max()

                print(f"BANK_ID: {bank_id}, RACK_ID: {rack_id}, PE: {partial_energy}")

                result_value[int(bank_id)][int(rack_id)] = {
                    "S_TIME": s_time,
                    "E_TIME": e_time,
                    "MIN_SOC": soc_min,
                    "MAX_SOC": soc_max,
                    "MIN_VOLTAGE": v_min,
                    "MAX_VOLTAGE": v_max,
                    "MIN_CURRENT": i_min,
                    "MAX_CURRENT": i_max,
                    "PE": partial_energy
                }

        return result_value
