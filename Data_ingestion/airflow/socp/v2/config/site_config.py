"""
사이트 설정 모듈

YAML 파일에서 사이트별 설정을 로드하고 관리하는 기능을 제공합니다.
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SiteConfig:
    """
    사이트별 설정을 담는 데이터 클래스

    Attributes:
        site_name: 사이트 이름 (예: 'baekma', 'gold', 'panly')
        db_conn_id: 사이트 데이터베이스 연결 ID
        stats_conn_id: 통계 데이터베이스 연결 ID
        tables: 테이블 이름 딕셔너리
        paths: 파일 경로 딕셔너리
        cell_count: 셀 개수
        file_format: 파일 포맷 ('parquet' 또는 'feather')
        period: 기간 설정
        query: 쿼리 관련 설정
        dag: DAG 관련 설정
    """
    site_name: str
    db_conn_id: str
    stats_conn_id: str
    tables: Dict[str, str]
    paths: Dict[str, str]
    cell_count: int
    file_format: str
    period: int
    query: Dict[str, Any]
    dag: Dict[str, Any]

    def __post_init__(self):
        """경로를 절대 경로로 확장"""
        for key, path in self.paths.items():
            self.paths[key] = os.path.expanduser(path)

    def get_table(self, table_type: str) -> str:
        """
        테이블 이름 조회

        Args:
            table_type: 테이블 타입 ('count', 'count_diff', 'info', 'moving_avg')

        Returns:
            테이블 이름
        """
        return self.tables.get(table_type, "")

    def get_path(self, path_type: str) -> str:
        """
        파일 경로 조회

        Args:
            path_type: 경로 타입 (예: 'original', 'count_work', 'count_prep' 등)

        Returns:
            파일 경로
        """
        return self.paths.get(path_type, "")

    def get_battery_status_columns(self) -> List[str]:
        """
        배터리 상태 컬럼 목록 조회

        Returns:
            배터리 상태 컬럼 리스트
        """
        return self.query.get("battery_status_columns", [])


class SiteConfigLoader:
    """
    사이트 설정을 YAML 파일에서 로드하는 클래스
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        초기화

        Args:
            config_path: 설정 파일 경로 (기본값: sites.yaml)
        """
        if config_path is None:
            # 현재 파일의 상위 디렉토리에서 sites.yaml 찾기
            current_dir = Path(__file__).parent.parent
            config_path = current_dir / "sites.yaml"

        self.config_path = config_path
        self._configs: Dict[str, SiteConfig] = {}
        self._load_configs()

    def _load_configs(self) -> None:
        """YAML 파일에서 설정 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            for site_name, site_data in data.get('sites', {}).items():
                self._configs[site_name] = SiteConfig(
                    site_name=site_name,
                    db_conn_id=site_data.get('db_conn_id', ''),
                    stats_conn_id=site_data.get('stats_conn_id', ''),
                    tables=site_data.get('tables', {}),
                    paths=site_data.get('paths', {}),
                    cell_count=site_data.get('cell_count', 240),
                    file_format=site_data.get('file_format', 'parquet'),
                    period=site_data.get('period', 1),
                    query=site_data.get('query', {}),
                    dag=site_data.get('dag', {}),
                )
        except FileNotFoundError:
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 파싱 오류: {e}")

    def get_config(self, site_name: str) -> SiteConfig:
        """
        특정 사이트의 설정 조회

        Args:
            site_name: 사이트 이름

        Returns:
            사이트 설정 객체

        Raises:
            KeyError: 존재하지 않는 사이트 이름인 경우
        """
        if site_name not in self._configs:
            raise KeyError(f"사이트 '{site_name}'의 설정을 찾을 수 없습니다. "
                         f"사용 가능한 사이트: {list(self._configs.keys())}")
        return self._configs[site_name]

    def get_all_sites(self) -> List[str]:
        """
        모든 사이트 이름 조회

        Returns:
            사이트 이름 리스트
        """
        return list(self._configs.keys())


# 싱글톤 인스턴스
_loader: Optional[SiteConfigLoader] = None


def get_site_config(site_name: str) -> SiteConfig:
    """
    사이트 설정을 조회하는 헬퍼 함수

    Args:
        site_name: 사이트 이름

    Returns:
        사이트 설정 객체

    Example:
        >>> config = get_site_config('baekma')
        >>> print(config.cell_count)
        241
    """
    global _loader
    if _loader is None:
        _loader = SiteConfigLoader()
    return _loader.get_config(site_name)
