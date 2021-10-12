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
import pandas as pd

if __name__ == "__main__":

    """데이터베이스 연결"""
    # CONNECTION = 
    conn = psycopg2.connect(CONNECTION)
    cursor = conn.cursor()

    # 데이터베이스 내 테이블리스트 조회
    sql = "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE'"
    cursor.execute(sql)
    result = cursor.fetchall()
    print(result)

    """데이터 조회"""
    # ESS 뱅크 데이터
    sql =  "SELECT * FROM bank LIMIT 100000000"
    cursor.execute(sql)
    result = pd.DataFrame(cursor.fetchall())
    print(result)

    # 데이터 크기 확인
    print(sys.getsizeof(result))