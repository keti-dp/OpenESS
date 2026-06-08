"""TIExVD (시간 간격 극값 전압 차이) 계산기"""

from typing import Dict, Any
import pandas as pd

from calculators.base_calculator import BaseHealthIndicatorCalculator


class TIExVDCalculator(BaseHealthIndicatorCalculator):
    """TIExVD 건강 지표 계산기"""

    def get_table_name(self) -> str:
        """데이터베이스 테이블 이름을 가져옵니다."""
        return "health_indicator_tiexvd"

    def get_insert_query(self) -> str:
        """SQL insert 쿼리 템플릿을 가져옵니다."""
        return """
            INSERT INTO health_indicator_tiexvd (
                "TIMESTAMP",
                "OPERATING_SITE",
                "BANK_ID",
                "RACK_ID",
                "TIECVD",
                "TIECVD_MIN_VOLTAGE",
                "TIECVD_MAX_VOLTAGE",
                "TIEDVD",
                "TIEDVD_MIN_VOLTAGE",
                "TIEDVD_MAX_VOLTAGE"
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

    def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[int, Dict[int, Any]]:
        """각 bank와 rack에 대해 TIExVD를 계산합니다.

        TIExVD는 극값 전압 간 시간 간격을 측정합니다:
        - TIECVD: 충전 전압 차이
        - TIEDVD: 방전 전압 차이

        Args:
            df: 배터리 데이터가 포함된 입력 DataFrame
            **kwargs: 추가 파라미터

        Returns:
            중첩된 dict: {bank_id: {rack_id: {metrics}}}
        """
        result_value = {}
        bank_list = df["BANK_ID"].unique()

        for bank_id in bank_list:
            result_value[int(bank_id)] = {}
            filtered_df = df.query(f"BANK_ID == {bank_id}")
            rack_list = filtered_df["RACK_ID"].unique()

            for rack_id in rack_list:
                try:
                    # 사이트의 배터리 상태 필드를 기반으로 충전 및 방전 데이터 쿼리
                    if 'BATTERY_STATUS_FOR_STANDBY' in filtered_df.columns:
                        # Gold/Panly 스타일
                        cvd_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_CHARGE == 1"
                        ).reset_index(drop=True)
                        dvd_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_DISCHARGE == 1"
                        ).reset_index(drop=True)
                    elif 'BATTERY_STATUS_OF_STAND_BY' in filtered_df.columns:
                        # Seokhwan 스타일
                        cvd_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_CHARGE == 1"
                        ).reset_index(drop=True)
                        dvd_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_DISCHARGE == 1"
                        ).reset_index(drop=True)
                    elif 'BATTERY_STATUS_FOR_RUN' in filtered_df.columns:
                        # Baekma/Seongdeok 스타일
                        cvd_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_RUN == 1 and BATTERY_STATUS_FOR_CHARGE == 2"
                        ).reset_index(drop=True)
                        dvd_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_RUN == 1 and BATTERY_STATUS_FOR_CHARGE == 3"
                        ).reset_index(drop=True)
                    else:
                        raise KeyError(
                            f"Required status columns not found. "
                            f"Available columns: {filtered_df.columns.tolist()}"
                        )

                    # TIECVD (Time Interval Extreme Charging Voltage Difference) 계산
                    cvd_min_idx = cvd_df["RACK_VOLTAGE"].idxmin()
                    cvd_max_idx = cvd_df["RACK_VOLTAGE"].idxmax()
                    cvd_time_diff = abs(
                        cvd_df.loc[cvd_min_idx, "TIMESTAMP"] -
                        cvd_df.loc[cvd_max_idx, "TIMESTAMP"]
                    ).total_seconds()

                    # TIEDVD (Time Interval Extreme Discharging Voltage Difference) 계산
                    dvd_min_idx = dvd_df["RACK_VOLTAGE"].idxmin()
                    dvd_max_idx = dvd_df["RACK_VOLTAGE"].idxmax()
                    dvd_time_diff = abs(
                        dvd_df.loc[dvd_min_idx, "TIMESTAMP"] -
                        dvd_df.loc[dvd_max_idx, "TIMESTAMP"]
                    ).total_seconds()

                    result_value[bank_id][int(rack_id)] = {
                        "cvd_time_diff": int(cvd_time_diff),
                        "cvd_min_vol": float(cvd_df["RACK_VOLTAGE"].min()),
                        "cvd_max_vol": float(cvd_df["RACK_VOLTAGE"].max()),
                        "dvd_time_diff": int(dvd_time_diff),
                        "dvd_min_vol": float(dvd_df["RACK_VOLTAGE"].min()),
                        "dvd_max_vol": float(dvd_df["RACK_VOLTAGE"].max())
                    }

                except ValueError:
                    print(f"오류: {bank_id}_{rack_id} TIExVD 계산에 필요한 데이터 없음")
                    continue

        return result_value
