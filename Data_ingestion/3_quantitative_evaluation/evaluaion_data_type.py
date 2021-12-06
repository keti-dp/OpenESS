# -*- coding:utf-8 -*-

"""
Copyright 2021, KETI.

2021-12-06 ver 1.0 evaluaion_data_type.py 

ESS 정량평가 중 수집 데이터 종류 평가를 위한 테스트 코드입니다. 

수집되는 데이터 종류 수를 측정합니다.

전체적인 코드에 대한 설명은 https://github.com/keti-dp/OpenESS 에서 확인하실 수 있습니다.
"""

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
