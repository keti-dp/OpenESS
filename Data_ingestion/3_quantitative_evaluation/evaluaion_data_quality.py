# -*- coding:utf-8 -*-

"""
evaluaion_data_quality.py : 대규모 분산 에너지 저장장치 과제 정량평가 테스트 코드

Copyright(C) 2021, 윤태일 / KETI / taeil777@keti.re.kr

최신 테스트 버전 : 1.0 ver
최신 안정화 버전 : 1.0 ver

ESS 정량평가 중 수집 데이터 품질 평가를 위한 테스트 코드입니다. 

수집되는 데이터 중 이상데이터 수에 따른 데이터 정확도를 측정합니다.

전체적인 코드에 대한 설명은 https://github.com/keti-dp/OpenESS 에서 확인하실 수 있습니다.
"""

import csv

if __name__ == "__main__":

    f = open(
        "csv파일경로",
        "r",
        encoding="utf-8",
    )
    rdr = csv.reader(f)

    count = 0
    next(f)  # 첫행삭제
    for line in rdr:
        for i in range(1, len(line)):
            if float(line[i]) > 6000:
                count += 1
                break
    print("이상 데이터 수 : ", count)

    f1 = open(
        "csv파일경로",
        "r",
        encoding="utf-8",
    )
    rdr = csv.reader(f1)

    # list 변경
    rdr_list = list(rdr)

    # 전체 행 수
    row_count = len(rdr_list)

    # 데이터 종류 수
    data_type_count = rdr_list[0]

    # 전체 데이터 수
    # print("전체 데이터 수 : ", row_count * len(data_type_count))
    print("전체 데이터 수 : ", row_count)

    # print("데이터 정확도 퍼센트 : ", 100 - (count / (row_count * len(data_type_count)) * 100))
    print("데이터 정확도 퍼센트 : ", 100 - (count / (row_count) * 100))

    f1.close()
