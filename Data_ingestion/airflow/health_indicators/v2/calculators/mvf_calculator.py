"""MVF (평균 전압 변동) 계산기"""

from typing import Dict, Any
import pandas as pd

from calculators.base_calculator import BaseHealthIndicatorCalculator


class MVFCalculator(BaseHealthIndicatorCalculator):
    """MVF 건강 지표 계산기"""

    def get_table_name(self) -> str:
        """데이터베이스 테이블 이름을 가져옵니다."""
        return "health_indicator_mvf"

    def get_insert_query(self) -> str:
        """SQL insert 쿼리 템플릿을 가져옵니다."""
        return """
            INSERT INTO health_indicator_mvf (
                "TIMESTAMP",
                "OPERATING_SITE",
                "BANK_ID",
                "RACK_ID",
                "MVF",
                "MIN_SOC",
                "MAX_SOC",
                "MIN_VOLTAGE",
                "MAX_VOLTAGE"
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

    def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[int, Dict[int, Any]]:
        """각 bank와 rack에 대해 MVF를 계산합니다.

        Args:
            df: 배터리 데이터가 포함된 입력 DataFrame
            **kwargs: 추가 파라미터

        Returns:
            중첩된 dict: {bank_id: {rack_id: {metrics}}}
        """
        nominal_rack_voltage = (
            self.site_config.nominal_cell_voltage *
            self.site_config.tray_count *
            12
        )

        result_value = {}
        bank_list = df["BANK_ID"].unique()

        for bank_id in bank_list:
            result_value[int(bank_id)] = {}
            filtered_df = df.query(f"BANK_ID == {bank_id}")
            rack_list = filtered_df["RACK_ID"].unique()

            for rack_id in rack_list:
                # 사이트의 배터리 상태 필드를 기반으로 데이터 쿼리
                if 'BATTERY_STATUS_FOR_STANDBY' in filtered_df.columns:
                    # Gold/Panly 스타일
                    mvf_df = filtered_df.query(
                        f"RACK_ID == {rack_id} and "
                        f"RACK_SOC >= {self.site_config.min_soc} and "
                        f"RACK_SOC <= {self.site_config.max_soc} and "
                        f"BATTERY_STATUS_FOR_DISCHARGE == 1"
                    ).reset_index(drop=True)
                elif 'BATTERY_STATUS_OF_STAND_BY' in filtered_df.columns:
                    # Seokhwan 스타일
                    mvf_df = filtered_df.query(
                        f"RACK_ID == {rack_id} and "
                        f"RACK_SOC >= {self.site_config.min_soc} and "
                        f"RACK_SOC <= {self.site_config.max_soc} and "
                        f"BATTERY_STATUS_FOR_DISCHARGE == 1"
                    ).reset_index(drop=True)
                elif 'BATTERY_STATUS_FOR_RUN' in filtered_df.columns:
                    # Baekma/Seongdeok 스타일
                    mvf_df = filtered_df.query(
                        f"RACK_ID == {rack_id} and "
                        f"RACK_SOC >= {self.site_config.min_soc} and "
                        f"RACK_SOC <= {self.site_config.max_soc} and "
                        f"BATTERY_STATUS_FOR_RUN == 1 and "
                        f"BATTERY_STATUS_FOR_CHARGE == 3"
                    ).reset_index(drop=True)
                else:
                    raise KeyError(
                        f"Required status columns not found. "
                        f"Available columns: {filtered_df.columns.tolist()}"
                    )

                mvf_df["RACK_VOLTAGE"] = abs(mvf_df["RACK_VOLTAGE"] - nominal_rack_voltage)

                try:
                    # SOC 범위가 유효한지 확인
                    if ((self.site_config.max_soc - 1) > mvf_df["RACK_SOC"].iloc[0]) or \
                       ((self.site_config.min_soc + 1) < mvf_df["RACK_SOC"].iloc[-1]):
                        average = float(0)
                    else:
                        average = round(float(mvf_df["RACK_VOLTAGE"].mean()), 2)

                    min_soc = float(mvf_df["RACK_SOC"].min())
                    max_soc = float(mvf_df["RACK_SOC"].max())
                    min_vol = round(float(mvf_df["RACK_VOLTAGE"].min()), 2)
                    max_vol = round(float(mvf_df["RACK_VOLTAGE"].max()), 2)

                except IndexError:
                    average = float(0)
                    min_soc = float(0)
                    max_soc = float(0)
                    min_vol = float(0)
                    max_vol = float(0)

                result_value[bank_id][int(rack_id)] = {
                    "mvf": average,
                    "min_soc": min_soc,
                    "max_soc": max_soc,
                    "min_vol": min_vol,
                    "max_vol": max_vol
                }

        return result_value
