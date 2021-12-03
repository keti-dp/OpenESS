# -*- coding:utf-8 -*-
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
