# -*- coding:utf-8 -*-

import psycopg2
from datetime import date, datetime
from psycopg2.extensions import AsIs
from dateutil.relativedelta import relativedelta
from pytz import timezone

import time
import threading
import os
from multiprocessing import Process

class MyProcess(Process):
    def __init__(self, string):
        Process.__init__(self)
        self.string = string

    def run(self):
        import_data(self.string)


class thread(threading.Thread):
    def __init__(self, file_path):
        threading.Thread.__init__(self)
        self.file_path = file_path

    def run(self):
        start = time.time()
        print("쓰레드 시작시간: ", start)
        import_data(self.file_path)
        end = time.time()
        print("쓰레드 처리시간: ", end - start)


def job(file_path):
    make_thread = thread(file_path)
    make_thread.start()


def import_data(file_path):

    conn = psycopg2.connect(
        host="1.1.1.1",
        dbname="ESS_Operating_Site1",
        user="postgres",
        password="#####",
        port="####",
    )
    # create a cursor object
    # cursor object is used to interact with the database
    cur = conn.cursor()

    # print("---DB 접속 확인---")
    # open the csv file using python standard file I/O
    # copy file into the table just created

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
        proc = MyProcess(
            "C:/Users/taeil/Documents/GitHub/ESS/Data_Ingestion/etc_202111011534.csv"
        )
        proc.start()

    end = time.time()

    print("전체 종료시간: ", end - start)