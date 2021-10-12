"""
title: 안전SW프레임워크 1차년도 성능시험평가
date: 2021-10-12
writer: KETI 최우정

    10.분석 및 학습을 위한 데이터 처리 크기
        -목표 달성 성능(GB/초)

        1)데이터 로드
            - 데이터 크기 확인
        2)데이터 처리
            - 평균 계산

"""

import sys
import psycopg2
import pandas.io.sql as psql
from datetime import datetime

if __name__ == "__main__":
    time_start = datetime.now()

    """데이터베이스 연결"""
    #CONNECTION =
    conn = psycopg2.connect(CONNECTION)
    cursor = conn.cursor()

    # 데이터베이스 내 테이블리스트 조회
    sql = "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE'"
    cursor.execute(sql)
    result = cursor.fetchall()
    print(result)

    """데이터 조회"""
    time_query_start = datetime.now()
    # ESS 뱅크 데이터
    sql =  "SELECT * FROM rack limit 100"
    df_rack = psql.read_sql(sql, conn)
    time_query_end = datetime.now()
    print(type(df_rack))
    print(df_rack)
    print('데이터 로드 시간:', time_query_end - time_query_start)

    # 데이터 크기 확인
    print(sys.getsizeof(df_rack))

    time_mean_start = datetime.now()
    # 평균 구하기
    print(df_rack.columns)
    rack_soh_mean = df_rack['RACK_SOH'].mean()
    print(rack_soh_mean)
    time_mean_end = datetime.now()
    print('평균 계산 시간:', time_mean_end - time_mean_start)
