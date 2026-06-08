"""DCIR (직류 내부 저항) 계산기"""

from typing import Dict, Any
import pandas as pd

from calculators.base_calculator import BaseHealthIndicatorCalculator


class DCIRCalculator(BaseHealthIndicatorCalculator):
    """DCIR 건강 지표 계산기"""

    def get_table_name(self) -> str:
        """데이터베이스 테이블 이름을 가져옵니다."""
        return "health_indicator_dcir"

    def get_insert_query(self) -> str:
        """SQL insert 쿼리 템플릿을 가져옵니다."""
        return """
            INSERT INTO health_indicator_dcir (
                "TIMESTAMP",
                "OPERATING_SITE",
                "BANK_ID",
                "RACK_ID",
                "CDCIR",
                "DDCIR"
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """

    def calculate(self, df: pd.DataFrame, **kwargs) -> Dict[int, Dict[int, Any]]:
        """각 bank와 rack에 대해 CDCIR 및 DDCIR을 계산합니다.

        Args:
            df: 배터리 데이터가 포함된 입력 DataFrame
            **kwargs: 추가 파라미터

        Returns:
            중첩된 dict: {bank_id: {rack_id: {"CDCIR": value, "DDCIR": value}}}
        """
        result_value = {}
        bank_list = df["BANK_ID"].unique()

        for bank_id in bank_list:
            result_value[int(bank_id)] = {}
            filtered_df = df.query(f"BANK_ID == {bank_id}")
            rack_list = filtered_df["RACK_ID"].unique()

            for rack_id in rack_list:
                try:
                    result_value[int(bank_id)][int(rack_id)] = {}

                    # 사이트의 배터리 상태 필드를 기반으로 데이터 쿼리
                    if 'BATTERY_STATUS_FOR_STANDBY' in filtered_df.columns:
                        # Gold/Panly 스타일
                        standby_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_STANDBY == 1"
                        ).reset_index(drop=True)
                        cdcir_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_CHARGE == 1"
                        ).reset_index(drop=True)
                        ddcir_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_DISCHARGE == 1"
                        ).reset_index(drop=True)
                    elif 'BATTERY_STATUS_OF_STAND_BY' in filtered_df.columns:
                        # Seokhwan 스타일
                        standby_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_OF_STAND_BY == 1"
                        ).reset_index(drop=True)
                        cdcir_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_CHARGE == 1"
                        ).reset_index(drop=True)
                        ddcir_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_DISCHARGE == 1"
                        ).reset_index(drop=True)
                    elif 'BATTERY_STATUS_FOR_RUN' in filtered_df.columns:
                        # Baekma/Seongdeok 스타일
                        standby_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_RUN == 1 and BATTERY_STATUS_FOR_CHARGE == 1"
                        ).reset_index(drop=True)
                        cdcir_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_RUN == 1 and BATTERY_STATUS_FOR_CHARGE == 2"
                        ).reset_index(drop=True)
                        ddcir_df = filtered_df.query(
                            f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_RUN == 1 and BATTERY_STATUS_FOR_CHARGE == 3"
                        ).reset_index(drop=True)
                    else:
                        # 필요한 상태 컬럼이 없는 경우 에러 발생
                        raise KeyError(
                            f"Required status columns not found. "
                            f"Available columns: {filtered_df.columns.tolist()}"
                        )

                    # TIMESTAMP를 index로 설정하고 중복 제거
                    standby_df = standby_df.set_index('TIMESTAMP')
                    cdcir_df = cdcir_df.set_index('TIMESTAMP')
                    ddcir_df = ddcir_df.set_index('TIMESTAMP')

                    standby_df = standby_df.loc[~standby_df.index.duplicated(keep='first')]
                    cdcir_df = cdcir_df.loc[~cdcir_df.index.duplicated(keep='first')]
                    ddcir_df = ddcir_df.loc[~ddcir_df.index.duplicated(keep='first')]

                    # CDCIR 계산
                    cdcir = self._calculate_cdcir(standby_df, cdcir_df)

                    # DDCIR 계산
                    ddcir = self._calculate_ddcir(standby_df, ddcir_df)

                    print(f"BANK_ID: {bank_id}, RACK_ID: {rack_id}, CDCIR: {cdcir}, DDCIR: {ddcir}")

                    result_value[int(bank_id)][int(rack_id)] = {
                        "CDCIR": cdcir,
                        "DDCIR": ddcir,
                    }

                except (ValueError, IndexError, KeyError) as e:
                    print(f"오류: {bank_id}_{rack_id} DCIR 계산에 필요한 데이터 없음: {e}")
                    continue

        return result_value

    def _calculate_cdcir(self, standby_df: pd.DataFrame, cdcir_df: pd.DataFrame) -> float:
        """충전 DCIR을 계산합니다.

        Args:
            standby_df: 대기 상태 DataFrame
            cdcir_df: 충전 상태 DataFrame

        Returns:
            CDCIR 값
        """
        min_time_cdcir = cdcir_df.index.min()
        charge_standby_df = standby_df[standby_df.index < min_time_cdcir]
        last_time_with_zero_current = charge_standby_df[charge_standby_df['RACK_CURRENT'] == 0].index[-1]

        filtered_cdcir_df = cdcir_df[
            (cdcir_df.index > last_time_with_zero_current) & (cdcir_df['RACK_CURRENT'] >= 10)
        ]
        closest_time_with_current_above_10 = filtered_cdcir_df.index.min()

        cdcir_v_diff = (
            cdcir_df.loc[closest_time_with_current_above_10, "RACK_VOLTAGE"] -
            standby_df.loc[last_time_with_zero_current, "RACK_VOLTAGE"]
        )
        cdcir_i_diff = (
            cdcir_df.loc[closest_time_with_current_above_10, "RACK_CURRENT"] -
            standby_df.loc[last_time_with_zero_current, "RACK_CURRENT"]
        )

        return round(abs(cdcir_v_diff / cdcir_i_diff), 2)

    def _calculate_ddcir(self, standby_df: pd.DataFrame, ddcir_df: pd.DataFrame) -> float:
        """방전 DCIR을 계산합니다.

        Args:
            standby_df: 대기 상태 DataFrame
            ddcir_df: 방전 상태 DataFrame

        Returns:
            DDCIR 값
        """
        min_time_ddcir = ddcir_df.index.min()
        discharge_standby_df = standby_df[standby_df.index < min_time_ddcir]
        last_time_with_zero_current = discharge_standby_df[discharge_standby_df['RACK_CURRENT'] == 0].index[-1]

        filtered_ddcir_df = ddcir_df[
            (ddcir_df.index > last_time_with_zero_current) & (ddcir_df['RACK_CURRENT'] <= -10)
        ]
        closest_time_with_current_above_10 = filtered_ddcir_df.index.min()

        ddcir_v_diff = (
            ddcir_df.loc[closest_time_with_current_above_10, "RACK_VOLTAGE"] -
            standby_df.loc[last_time_with_zero_current, "RACK_VOLTAGE"]
        )
        ddcir_i_diff = (
            ddcir_df.loc[closest_time_with_current_above_10, "RACK_CURRENT"] -
            standby_df.loc[last_time_with_zero_current, "RACK_CURRENT"]
        )

        return round(abs(ddcir_v_diff / ddcir_i_diff), 2)
