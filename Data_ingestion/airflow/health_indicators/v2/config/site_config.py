"""ESS 건강 지표 처리를 위한 사이트별 설정"""

import os
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime
import pendulum


@dataclass
class SiteConfig:
    """특정 ESS 사이트에 대한 설정"""

    site_id: int
    site_name: str
    postgres_conn_id: str
    data_path: str
    schedule_interval: str
    start_date: datetime
    battery_status_fields: List[str]

    # 배터리 사양
    nominal_cell_voltage: float = 4.2
    tray_count: int = 20

    # 건강 지표 파라미터
    max_soc: int = 65
    min_soc: int = 40
    diff_soc: int = 3


# 사이트 설정
SITES = {
    'seokhwan': SiteConfig(
        site_id=6,
        site_name='seokhwan',
        postgres_conn_id='seokhwan_site',
        data_path=os.path.join(os.path.expanduser("~"), "airflow/dags/calc_keti_health_indicator/dataset/seokhwan/"),
        schedule_interval='00 05 * * *',
        start_date=datetime(2025, 11, 6, tzinfo=pendulum.timezone("Asia/Seoul")),
        battery_status_fields=['BATTERY_STATUS_OF_STAND_BY', 'BATTERY_STATUS_FOR_CHARGE', 'BATTERY_STATUS_FOR_DISCHARGE']
    ),

    'seongdeok': SiteConfig(
        site_id=5,
        site_name='seongdeok',
        postgres_conn_id='seongdeok_site',
        data_path=os.path.join(os.path.expanduser("~"), "airflow/dags/calc_keti_health_indicator/dataset/seongdeok/"),
        schedule_interval='00 05 * * *',
        start_date=datetime(2025, 11, 6, tzinfo=pendulum.timezone("Asia/Seoul")),
        battery_status_fields=['BATTERY_STATUS_FOR_RUN', 'BATTERY_STATUS_FOR_CHARGE']
    ),

    'baekma': SiteConfig(
        site_id=4,
        site_name='baekma',
        postgres_conn_id='baekma_site',
        data_path=os.path.join(os.path.expanduser("~"), "airflow/dags/calc_keti_health_indicator/dataset/baekma/"),
        schedule_interval='00 05 * * *',
        start_date=datetime(2025, 11, 6, tzinfo=pendulum.timezone("Asia/Seoul")),
        battery_status_fields=['BATTERY_STATUS_FOR_RUN', 'BATTERY_STATUS_FOR_CHARGE']
    ),
    'gold': SiteConfig(
        site_id=3,
        site_name='gold',
        postgres_conn_id='gold_site',
        data_path=os.path.join(os.path.expanduser("~"), "airflow/dags/calc_keti_health_indicator/dataset/gold/"),
        schedule_interval='00 05 * * *',
        start_date=datetime(2025, 11, 6, tzinfo=pendulum.timezone("Asia/Seoul")),
        battery_status_fields=['BATTERY_STATUS_FOR_STANDBY', 'BATTERY_STATUS_FOR_CHARGE', 'BATTERY_STATUS_FOR_DISCHARGE']
    ),
    'panly': SiteConfig(
        site_id=2,
        site_name='panly',
        postgres_conn_id='panly_site',
        data_path=os.path.join(os.path.expanduser("~"), "airflow/dags/calc_keti_health_indicator/dataset/panly/"),
        schedule_interval='00 05 * * *',
        start_date=datetime(2025, 11, 6, tzinfo=pendulum.timezone("Asia/Seoul")),
        battery_status_fields=['BATTERY_STATUS_FOR_STANDBY', 'BATTERY_STATUS_FOR_CHARGE', 'BATTERY_STATUS_FOR_DISCHARGE']
    )
}


# 건강 지표 설정
HEALTH_INDICATORS = ['DCIR', 'MVF', 'PE', 'TIExVD', 'VIExTD']


# 공통 DAG 기본 인자
DEFAULT_DAG_ARGS = {
    'owner': 'jwpark',
    'catchup': True
}


def get_site_config(site_name: str) -> SiteConfig:
    """특정 사이트의 설정을 가져옵니다.

    Args:
        site_name: 사이트 이름 (baekma, gold, panly, seongdeok, seokhwan)

    Returns:
        지정된 사이트에 대한 SiteConfig 객체

    Raises:
        ValueError: site_name을 찾을 수 없는 경우
    """
    if site_name not in SITES:
        raise ValueError(f"알 수 없는 사이트: {site_name}. 사용 가능한 사이트: {list(SITES.keys())}")
    return SITES[site_name]
