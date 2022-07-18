# 데이터 규격에 따른 데이터 업데이트
# 특정 기간동안, 특정 테이블의, 특정 컬럼의 값을 수정할 때 사용할 모듈

# ex) 2022 03 25 ~ 2022 06 01 까지, GCP의 Operation1 에서 bank 테이블의, Rack_status_for_Run 의 값이 0이면 1로 1이면 0으로 수정

import pytz
import timescale_input_test
from pytz import timezone
from datetime import date, datetime, timedelta
from datetime import datetime
import logs
import pandas as pd
import sys


if __name__ == "__main__":

    # 0. 기간설정
    seoul = pytz.timezone("Asia/Seoul")
    start_time = datetime(2022, 2, 18, 0, 0, 0)
    start_time = seoul.localize(start_time)
    end_time = datetime(2022, 3, 26, 0, 0, 0)
    end_time = seoul.localize(end_time)

    # 1. DB설정
    operating_site = "dbname"

    timescale_db = timescale_input_test.timescale(
        ip="ip",
        port="port",
        username="username",
        password="password",
        dbname=operating_site,
    )

    # 2. 단순 값 변경
    # table_name = "rack"
    # set_column = "RACK_STATUS_FOR_RUN"
    # set_value = 1
    # target_value = 0

    # query_message = """UPDATE {table_name} SET "{set_column}" = {set_value} WHERE ("TIMESTAMP"
    # between '{start_time}' and '{end_time}') and "{set_column}" = {target_value};""".format(
    #     table_name=table_name,
    #     set_column=set_column,
    #     set_value=set_value,
    #     target_value=target_value,
    #     start_time=start_time,
    #     end_time=end_time,
    # )

    # 3. scalefactor 변경
    table_name = "etc"
    set_column = "SENSOR1_HUMIDITY"
    target_value = 10

    query_message = """UPDATE {table_name} SET "{set_column}" = "{set_column}" * {target_value} WHERE ("TIMESTAMP" 
    between '{start_time}' and '{end_time}') and "{set_column}" < {target_value};""".format(
        table_name=table_name,
        set_column=set_column,
        target_value=target_value,
        start_time=start_time,
        end_time=end_time,
    )

    # query_message = "select * from etc limit 10;"

    timescale_db.query_data(query_message)
