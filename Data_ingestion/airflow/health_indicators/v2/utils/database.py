"""데이터베이스 유틸리티 함수"""

from typing import Dict, Any
from airflow.providers.postgres.hooks.postgres import PostgresHook

from calculators.base_calculator import BaseHealthIndicatorCalculator


def push_health_indicator_to_database(
    calculator: BaseHealthIndicatorCalculator,
    task_id: str,
    **context
) -> None:
    """건강 지표 결과를 데이터베이스에 push합니다.

    Args:
        calculator: 건강 지표 계산기 인스턴스
        task_id: 결과를 가져올 Task ID
        **context: Airflow context
    """
    result = context["task_instance"].xcom_pull(task_ids=task_id, key="result")
    query_time = context["task_instance"].xcom_pull(task_ids="calc_time_range", key="query_time")
    operating_site = context["task_instance"].xcom_pull(task_ids="initialize_globals", key="OPERATING_SITE")

    pg_hook = PostgresHook(postgres_conn_id='ess_stats')
    conn = pg_hook.get_conn()
    cur = conn.cursor()

    insert_query = calculator.get_insert_query()

    try:
        for bank_id, racks in result.items():
            for rack_id, values in racks.items():
                if not values:
                    continue

                print(f"삽입 중: bank_id={bank_id}, rack_id={rack_id}, values={values}")

                # 결과의 값을 기반으로 파라미터 튜플 생성
                params = build_insert_params(
                    query_time["start_time"],
                    operating_site,
                    bank_id,
                    rack_id,
                    values,
                    calculator
                )

                try:
                    cur.execute(insert_query, params)
                    conn.commit()
                except KeyError as e:
                    print(f"오류: {bank_id}_{rack_id} 누락된 키: {e}")
                    continue

    finally:
        cur.close()
        conn.close()


def build_insert_params(
    timestamp: str,
    operating_site: int,
    bank_id: int,
    rack_id: int,
    values: Dict[str, Any],
    calculator: BaseHealthIndicatorCalculator
) -> tuple:
    """INSERT 쿼리를 위한 파라미터 튜플을 생성합니다.

    Args:
        timestamp: 타임스탬프 문자열
        operating_site: 운영 사이트 ID
        bank_id: Bank ID
        rack_id: Rack ID
        values: 메트릭 값 딕셔너리
        calculator: 계산기 인스턴스 (타입 확인용)

    Returns:
        INSERT 쿼리를 위한 파라미터 튜플
    """
    from calculators.dcir_calculator import DCIRCalculator
    from calculators.mvf_calculator import MVFCalculator
    from calculators.pe_calculator import PECalculator
    from calculators.tiexvd_calculator import TIExVDCalculator
    from calculators.viextd_calculator import VIExTDCalculator

    base = (timestamp, operating_site, bank_id, rack_id)

    if isinstance(calculator, DCIRCalculator):
        return base + (values["CDCIR"], values["DDCIR"])

    elif isinstance(calculator, MVFCalculator):
        return base + (
            values["mvf"],
            values["min_soc"],
            values["max_soc"],
            values["min_vol"],
            values["max_vol"]
        )

    elif isinstance(calculator, PECalculator):
        return base + (
            values["S_TIME"],
            values["E_TIME"],
            values["MIN_SOC"],
            values["MAX_SOC"],
            values["MIN_VOLTAGE"],
            values["MAX_VOLTAGE"],
            values["MIN_CURRENT"],
            values["MAX_CURRENT"],
            values["PE"]
        )

    elif isinstance(calculator, TIExVDCalculator):
        return base + (
            values["cvd_time_diff"],
            values["cvd_min_vol"],
            values["cvd_max_vol"],
            values["dvd_time_diff"],
            values["dvd_min_vol"],
            values["dvd_max_vol"]
        )

    elif isinstance(calculator, VIExTDCalculator):
        return base + (
            values["S_VIECTD"],
            values["E_VIECTD"],
            values["VIECTD"],
            values["S_VIEDTD"],
            values["E_VIEDTD"],
            values["VIEDTD"]
        )

    else:
        raise ValueError(f"알 수 없는 계산기 타입: {type(calculator)}")
