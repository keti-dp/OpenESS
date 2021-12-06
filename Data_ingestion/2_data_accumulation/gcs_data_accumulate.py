#!/bin/python

"""
Copyright 2021, KETI.

2021-12-06 ver 1.0 gcs_data_accumulate.py 

timescaleDB에 저장된 데이터를 gcs에 축적하는 코드입니다.

수집된 데이터를 시간에 맞춰 csv 파일로 저장하고 

저장된 csv파일을 gcs 버킷에 날짜별로 업로드합니다.

전체적인 코드에 대한 설명은 https://github.com/keti-dp/OpenESS 에서 확인하실 수 있습니다.
"""


from datetime import datetime
import datetime
from pytz import timezone
from multiprocessing import Process
from pytz import timezone
import numpy as np
import timescale_input_data
import csv
import os
import gcp_storage
import logs

# 구글 클라우드 스토리지 클래스
class GCS:
    pass


# csv 추출 메서드
def csv_export(tablename):

    # 로그
    export_logger = logs.get_logger("log1", "/log/", "csv_export.log")

    # 현재시간계산
    current = datetime.datetime.now()

    # 현재시간 / 전날시간
    current_time = current.replace(hour=0, minute=0, second=0, microsecond=0)
    previous_time = current_time + datetime.timedelta(days=-1)

    year = current_time.year
    year2 = "{:%y}".format(current_time)
    month = "{:%m}".format(current_time)
    day = "{:%d}".format(current_time)

    previous_year = previous_time.year
    previous_year2 = "{:%y}".format(previous_time)
    previous_month = "{:%m}".format(previous_time)
    yesterday = "{:%d}".format(previous_time)

    query = """SET TIME ZONE 'Asia/Seoul';
            select * from "{_table}" where "TIMESTAMP" between '{_previoustime}' and '{_currenttime}' order by "TIMESTAMP" """.format(
        _table=tablename,
        _currenttime=current_time,
        _previoustime=previous_time,
    )

    rows = timescale_test.query_data(query)

    fields_query = """select column_name from information_schema.columns where table_catalog = 'ESS_Operating_Site1' and table_name = '{_table}' order by ordinal_position;""".format(
        _table=tablename
    )
    fields = timescale_test.query_data(fields_query)
    fields_list = []

    for row in fields:
        fields_list.append(row[0])

    file_path = """/csvfiles/{_year}/{_month}/{_day}/""".format(
        _year=previous_year, _month=previous_month, _day=yesterday
    )

    file_name = """{_year}{_month}{_day}_{_tablename}.csv""".format(
        _year=previous_year,
        _month=previous_month,
        _day=yesterday,
        _tablename=tablename,
    )

    try:
        os.makedirs(file_path)
    except OSError:
        if not os.path.isdir(file_path):
            raise

    try:
        with open(file_path + file_name, "w", newline="") as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(fields_list)
            write.writerows(rows)

        export_logger.info("%s csv export success", file_name)
        print("----- csv export -----")

    except Exception as e:
        export_logger.error("%s csv export error : ", file_name, e)
    # csv 파일 만들기


def csv_upload(tablename):

    # 로그
    upload_logger = logs.get_logger("log2", "/log/", "csv_upload.log")

    # 현재시간계산
    current = datetime.datetime.now()

    # 현재시간 / 전날시간
    current_time = current.replace(hour=0, minute=0, second=0, microsecond=0)
    previous_time = current_time + datetime.timedelta(days=-1)

    year = current_time.year
    month = "{:%m}".format(current_time)
    day = "{:%d}".format(current_time)
    yesterday = "{:%d}".format(previous_time)

    previous_year = previous_time.year
    previous_year2 = "{:%y}".format(previous_time)
    previous_month = "{:%m}".format(previous_time)
    yesterday = "{:%d}".format(previous_time)

    file_path = """/csvfiles/{_year}/{_month}/{_day}/{_year}{_month}{_day}_{_tablename}.csv""".format(
        _year=previous_year2,
        _month=previous_month,
        _day=yesterday,
        _tablename=tablename,
    )

    file_name = (
        """{_year}/{_month}/{_day}/{_year}{_month}{_day}_{_tablename}.csv""".format(
            _year=previous_year,
            _month=previous_month,
            _day=yesterday,
            _tablename=tablename,
        )
    )

    file_name2 = """{_year}{_month}{_day}_{_tablename}.csv""".format(
        _year=previous_year,
        _month=previous_month,
        _day=yesterday,
        _tablename=tablename,
    )

    try:
        gcp_storage.upload_2bucket(file_name, file_path)

        upload_logger.info("%s csv upload success", file_name2)

    except Exception as e:
        upload_logger.error("%s csv upload error", file_name2, e)


if __name__ == "__main__":

    timescale_test = timescale_input_data.timescale()

    csv_export("bank")
    csv_export("rack")
    csv_export("pcs")
    csv_export("etc")

    csv_upload("bank")
    csv_upload("rack")
    csv_upload("pcs")
    csv_upload("etc")
