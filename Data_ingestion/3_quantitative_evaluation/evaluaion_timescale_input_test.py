# -*- coding:utf-8 -*-

"""
Copyright 2021, KETI.

2021-12-06 ver 1.0 evaluaion_timescale_input_test.py 

ESS 정량평가 중 데이터 처리 속도 평가를 위한 테스트 코드입니다. 

데이터 저장 시 소요되는 시간을 측정합니다.

전체적인 코드에 대한 설명은 https://github.com/keti-dp/OpenESS 에서 확인하실 수 있습니다.
"""

import psycopg2
from datetime import date, datetime
from psycopg2.extensions import AsIs
from dateutil.relativedelta import relativedelta
from pytz import timezone

import time
from multiprocessing import Process


class MyProcess(Process):
    def __init__(self, string):
        Process.__init__(self)
        self.string = string

    def run(self):

        import_data(self.string)


def import_data(file_path):

    conn = psycopg2.connect(
        host="ip주소",
        dbname="db명",
        user="username",
        password="password",
        port="port번호",
    )
    # create a cursor object
    # cursor object is used to interact with the database
    cur = conn.cursor()

    with open(
        file_path,
        "r",
    ) as f:
        # print("---파일 open---")
        next(f)  # Skip the header row.
        # f , <database name>, Comma-Seperated

        start = time.time()

        cur.copy_from(f, "etc_1", sep=",")
        # print("---파일 입력---")
        # Commit Changes

        conn.commit()
        # print("---변경사항 저장---")
        end = time.time()
        print("프로세스 종료시간: ", end - start)

        # Close connection
        conn.close()
    f.close()


if __name__ == "__main__":

    start = time.time()

    for i in range(20):
        proc = MyProcess("csv파일경로")
        proc.start()

    end = time.time()

    print("전체 종료시간: ", end - start)
