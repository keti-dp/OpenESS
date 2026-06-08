"""
시간 계산 유틸리티 모듈

Airflow execution_date를 기반으로 한 시간 계산 기능을 제공합니다.
"""

from datetime import datetime, timedelta
from typing import Dict, Any
from pytz import timezone
import pendulum


class TimeCalculator:
    """
    시간 계산을 위한 유틸리티 클래스
    """

    def __init__(self, tz: str = "Asia/Seoul"):
        """
        초기화

        Args:
            tz: 타임존 (기본값: Asia/Seoul)
        """
        self.tz = timezone(tz)

    def convert_to_seoul(self, dt: datetime) -> pendulum.DateTime:
        """
        UTC 시간을 서울 시간으로 변환

        Args:
            dt: UTC datetime 객체

        Returns:
            서울 시간대로 변환된 pendulum DateTime 객체
        """
        if isinstance(dt, pendulum.DateTime):
            return dt.astimezone(self.tz)
        # datetime을 pendulum으로 변환
        if dt.tzinfo is None:
            dt = pendulum.instance(dt, tz='UTC')
        else:
            dt = pendulum.instance(dt)
        return dt.astimezone(self.tz)

    def calc_past_days_for_diff(self, execution_date: datetime) -> Dict[str, str]:
        """
        1st differencing용 시간 범위 계산
        전날부터 실행일 시작까지의 범위를 반환

        Args:
            execution_date: Airflow execution_date

        Returns:
            시작 시간과 종료 시간을 담은 딕셔너리
            - start: 전날 00:00:00
            - end: 실행일 00:00:00
        """
        seoul_time = self.convert_to_seoul(execution_date)

        start = (seoul_time - timedelta(days=1)).start_of('day').to_datetime_string()
        end = seoul_time.start_of('day').to_datetime_string()

        return {"start": start, "end": end}

    def calc_day_range(self, execution_date: datetime) -> Dict[str, str]:
        """
        해당 날짜의 시작과 끝 시간 계산
        실행일 00:00:00부터 23:59:59까지

        Args:
            execution_date: Airflow execution_date

        Returns:
            시작 시간과 종료 시간을 담은 딕셔너리
            - begin_time: 실행일 00:00:00
            - end_time: 실행일 23:59:59
        """
        seoul_time = self.convert_to_seoul(execution_date)

        begin_time = seoul_time.start_of('day').to_datetime_string()
        end_time = seoul_time.end_of('day').to_datetime_string()

        return {"begin_time": begin_time, "end_time": end_time}

    def calc_period_ranges(self, execution_date: datetime) -> Dict[str, str]:
        """
        여러 기간의 시간 범위 계산 (1일, 7일, 30일)

        Args:
            execution_date: Airflow execution_date

        Returns:
            각 기간의 시작 시간을 담은 딕셔너리
            - past_1days: 실행일 00:00:00
            - past_7days: 6일 전 00:00:00
            - past_30days: 29일 전 00:00:00
            - end_time: 실행일 23:59:59
        """
        seoul_time = self.convert_to_seoul(execution_date)

        past_1days = seoul_time.start_of('day').to_datetime_string()
        past_7days = (seoul_time - pendulum.duration(days=6)).start_of('day').to_datetime_string()
        past_30days = (seoul_time - pendulum.duration(days=29)).start_of('day').to_datetime_string()
        end_time = seoul_time.end_of('day').to_datetime_string()

        return {
            "past_30days": past_30days,
            "past_7days": past_7days,
            "past_1days": past_1days,
            "end_time": end_time,
        }

    @staticmethod
    def format_filename(dt_string: str) -> str:
        """
        날짜 문자열을 파일명 형식으로 변환

        Args:
            dt_string: 날짜 문자열 (예: "2024-12-11 00:00:00")

        Returns:
            파일명 형식의 문자열 (예: "20241211")
        """
        return dt_string[:4] + dt_string[5:7] + dt_string[8:10]

    @staticmethod
    def format_date(dt_string: str) -> str:
        """
        날짜 문자열에서 날짜 부분만 추출

        Args:
            dt_string: 날짜 문자열 (예: "2024-12-11 00:00:00")

        Returns:
            날짜 문자열 (예: "2024-12-11")
        """
        return dt_string[:10]
