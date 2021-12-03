# -*- coding:utf-8 -*-

import psycopg2
import time
import csv
import pprint

if __name__ == "__main__":

    start = time.time()
    f = open(
        "csv파일경로",
        "r",
        encoding="utf-8",
    )
    rdr = csv.reader(f)
    for line in rdr:
        pprint(line)
        print("수집되는 데이터 종류 수 : ", len(line))
        break
    f.close()

    end = time.time()
    print("전체 종료시간: ", end - start)
