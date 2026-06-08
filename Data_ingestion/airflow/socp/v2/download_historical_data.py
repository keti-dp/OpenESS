"""
Historical Data Download Script

새로운 사이트의 과거 30일치 데이터를 다운로드하는 스크립트입니다.
SoCP 알고리즘 수행을 위한 초기 데이터 준비용입니다.

사용법:
    python download_historical_data.py <site_name> [days]

예시:
    python download_historical_data.py oper5 30
    python download_historical_data.py oper6 30
"""

import sys
import os
from datetime import datetime, timedelta
import pendulum

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_site_config
from processors import DatasetProcessor
from core import TimeCalculator


def download_historical_data(site_name: str, days: int = 30):
    """
    지정된 사이트의 과거 데이터를 다운로드합니다.

    Args:
        site_name: 사이트 이름 (예: 'oper5', 'oper6')
        days: 다운로드할 일수 (기본값: 30일)
    """
    print(f"=" * 60)
    print(f"Historical Data Download for {site_name}")
    print(f"=" * 60)

    try:
        # 사이트 설정 로드
        config = get_site_config(site_name)
        print(f"✓ 사이트 설정 로드 완료: {site_name}")
        print(f"  - DB Connection: {config.db_conn_id}")
        print(f"  - Cell Count: {config.cell_count}")
        print(f"  - File Format: {config.file_format}")

    except Exception as e:
        print(f"✗ 사이트 설정을 찾을 수 없습니다: {site_name}")
        print(f"  Error: {e}")
        print(f"\n sites.yaml에 {site_name} 설정을 추가해주세요.")
        return False

    # 데이터셋 프로세서 초기화
    processor = DatasetProcessor(config)
    time_calc = TimeCalculator()

    # 오늘 날짜 기준으로 과거 N일 계산
    kst = pendulum.timezone("Asia/Seoul")
    today = datetime.now(kst)

    print(f"\n다운로드 기간: {days}일")
    print(f"종료 날짜: {today.strftime('%Y-%m-%d')}")
    print(f"시작 날짜: {(today - timedelta(days=days-1)).strftime('%Y-%m-%d')}")
    print(f"\n" + "=" * 60)

    # 각 날짜별로 데이터 다운로드
    success_count = 0
    fail_count = 0

    for i in range(days - 1, -1, -1):  # 과거부터 현재까지
        target_date = today - timedelta(days=i)

        # 해당 날짜의 시작/종료 시간 계산
        begin_time = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = begin_time + timedelta(days=1) - timedelta(seconds=1)

        begin_time_str = begin_time.strftime('%Y-%m-%d %H:%M:%S')
        end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')

        print(f"\n[{days - i}/{days}] {target_date.strftime('%Y-%m-%d')} 데이터 다운로드 중...")

        try:
            filename = processor.fetch_and_save_dataset(begin_time_str, end_time_str)

            if filename:
                print(f"  ✓ 성공: {filename}")
                success_count += 1
            else:
                print(f"  ✗ 실패: 데이터 없음")
                fail_count += 1

        except Exception as e:
            print(f"  ✗ 오류 발생: {e}")
            fail_count += 1

    # 결과 요약
    print(f"\n" + "=" * 60)
    print(f"다운로드 완료!")
    print(f"  성공: {success_count}개")
    print(f"  실패: {fail_count}개")
    print(f"  총계: {days}개")
    print(f"=" * 60)

    return success_count > 0


def main():
    """메인 실행 함수"""
    if len(sys.argv) < 2:
        print("사용법: python download_historical_data.py <site_name> [days]")
        print("\n예시:")
        print("  python download_historical_data.py oper5 30")
        print("  python download_historical_data.py oper6 30")
        print("\n사용 가능한 사이트:")

        # 사용 가능한 사이트 목록 표시
        try:
            from config.site_config import SiteConfigLoader
            loader = SiteConfigLoader()
            sites = loader.get_all_sites()
            for site in sites:
                print(f"  - {site}")
        except Exception as e:
            print(f"  (사이트 목록을 가져올 수 없습니다: {e})")

        sys.exit(1)

    site_name = sys.argv[1]
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    # 데이터 다운로드 실행
    success = download_historical_data(site_name, days)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
