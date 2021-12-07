# -*- coding:utf-8 -*-

"""
evaluaion_data_type.py : 대규모 분산 에너지 저장장치 과제 정량평가 테스트 코드

Copyright(C) 2021, 윤태일 / KETI / taeil777@keti.re.kr

최신 테스트 버전 : 1.0 ver
최신 안정화 버전 : 1.0 ver

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
