from contextlib import nullcontext
from numpy import e
import psycopg2
from datetime import date, datetime
from pytz import timezone
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.sql.expression import false


class timescale:

    # 기본 클라이언트 설정
    def __init__(self, ip, port, username, password, dbname):
        # timescale DB 연결
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.dbname = dbname

        self.CONNECTION = (
            """postgres://{_username}:{_password}@{_ip}:{_port}/{_dbname}""".format(
                _username=username,
                _password=password,
                _ip=ip,
                _port=port,
                _dbname=dbname,
            )
        )
        with psycopg2.connect(self.CONNECTION) as self.conn:
            self.cursor = self.conn.cursor()

        print("---------timescaledb connected----------")

    def create_hypertable(self, table_name):

        # Bank 생성 SQL문
        query_create_sensordata_table = """CREATE TABLE "{_tablename}" (
                                                "TIMESTAMP" timestamptz NOT NULL,
                                                "ERROR_CODE" int4 NOT NULL,
                                                "LEVEL" int4 NULL,
                                                "BANK_ID" int4 NOT NULL,
                                                "RACK_ID" int4 NOT NULL
                                            );""".format(
            _tablename=table_name
        )

        #    FOREIGN KEY (Bank_ID) REFERENCES RACK (Bank_ID)

        query_create_sensordata_hypertable = (
            """SELECT create_hypertable('{_tablename}', 'TIMESTAMP');""".format(
                _tablename=table_name
            )
        )

        self.cursor.execute(query_create_sensordata_table)

        self.cursor.execute(query_create_sensordata_hypertable)
        # commit changes to the database to make changes persistent
        self.conn.commit()
        self.cursor.close()

    def query_data(self, query_text):
        cursor = self.conn.cursor()
        # query = """SELECT "TIMESTAMP" FROM public.bank;"""
        cursor.execute(query_text)
        result = cursor.fetchall()
        cursor.close()
        return result

    def create_engine(self):

        test = (
            """postgresql://{_username}:{_password}@{_ip}:{_port}/{_dbname}""".format(
                _username=self.username,
                _password=self.password,
                _ip=self.ip,
                _port=self.port,
                _dbname=self.dbname,
            )
        )
        print(test)
        engine = create_engine(test)
        return engine


# 과충전
def protectionmap_1(max_cell_voltage):
    if max_cell_voltage > 4.10:
        result = 2
    elif max_cell_voltage > 4.05:
        result = 1
    else:
        result = 0
    return result


# 과방전
def protectionmap_2(min_cell_voltage):
    if min_cell_voltage < 3.00:
        result = 2
    elif min_cell_voltage < 3.20:
        result = 1
    else:
        result = 0
    return result


# 전압불평형
def protectionmap_3(max_cell_voltage, min_cell_voltage):
    voltage_gap = max_cell_voltage - min_cell_voltage

    if voltage_gap > 0.5:
        result = 2
    elif voltage_gap > 0.3:
        result = 1
    else:
        result = 0
    return result


# 충전 과전류
def protectionmap_4(charging_status, current):

    if charging_status == 1 and current > 120:
        result = 2
    else:
        result = 0
    return result


# 방전 과전류
def protectionmap_5(discharging_status, current):
    if discharging_status == 1 and current > 120:
        result = 2
    else:
        result = 0
    return result


# 고온
def protectionmap_6(max_temp):
    if max_temp > 55:
        result = 2
    elif max_temp > 50:
        result = 1
    else:
        result = 0
    return result


# 저온
def protectionmap_7(min_temp):
    if min_temp < -10:
        result = 2
    elif min_temp < 0:
        result = 1
    else:
        result = 0
    return result


# 온도불평형
def protectionmap_8(max_temp, min_temp):
    temp_gap = max_temp - min_temp
    if temp_gap > 25:
        result = 2
    elif temp_gap > 20:
        result = 1
    else:
        result = 0
    return result


if __name__ == "__main__":
    test1 = timescale(
        ip="",
        port="",
        username="",
        password="",
        dbname="",
    )

    # Operating Site 정보 입력
    test2 = timescale(
        ip="",
        port="",
        username="",
        password="",
        dbname="",
    )

    result = test2.query_data(
        """select 
        "TIMESTAMP", 
        "BANK_ID",
        "RACK_ID",
        "RACK_MAX_CELL_VOLTAGE",
        "RACK_MIN_CELL_VOLTAGE",        
        "RACK_CURRENT_SENSOR_CHARGE",
        "RACK_CURRENT_SENSOR_DISCHARGE",
        "RACK_CURRENT", 
        "RACK_MAX_CELL_TEMPERATURE",
        "RACK_MIN_CELL_TEMPERATURE"
        from rack r where "TIMESTAMP" between now() - interval '1'minute and now();"""
    )

    data_df = pd.DataFrame(result)
    data_df.columns = [
        "TIMESTAMP",
        "BANK_ID",
        "RACK_ID",
        "RACK_MAX_CELL_VOLTAGE",
        # "RACK_MAX_CELL_VOLTAGE_POSITION",
        "RACK_MIN_CELL_VOLTAGE",
        # "RACK_MIN_CELL_VOLTAGE_POSITION",
        "RACK_CURRENT_SENSOR_CHARGE",
        "RACK_CURRENT_SENSOR_DISCHARGE",
        "RACK_CURRENT",
        "RACK_MAX_CELL_TEMPERATURE",
        # "RACK_MAX_CELL_TEMPERATURE_POSITION",
        "RACK_MIN_CELL_TEMPERATURE",
        # "RACK_MIN_CELL_TEMPERATURE_POSITION",
    ]
    print(data_df)

    result_df = pd.DataFrame()

    result_df["TIMESTAMP"] = data_df["TIMESTAMP"]
    result_df["BANK_ID"] = data_df["BANK_ID"]
    result_df["RACK_ID"] = data_df["RACK_ID"]

    # df["value"].apply(lambda v: get_score(v))

    result_df["code1"] = data_df["RACK_MAX_CELL_VOLTAGE"].apply(
        lambda v: protectionmap_1(v)
    )
    result_df["code2"] = data_df["RACK_MIN_CELL_VOLTAGE"].apply(
        lambda v: protectionmap_2(v)
    )

    result_df["code3"] = data_df[
        ["RACK_MAX_CELL_VOLTAGE", "RACK_MIN_CELL_VOLTAGE"]
    ].apply(lambda v: protectionmap_3(v[0], v[1]), axis=1)

    result_df["code4"] = data_df[["RACK_CURRENT_SENSOR_CHARGE", "RACK_CURRENT"]].apply(
        lambda v: protectionmap_4(v[0], v[1]), axis=1
    )

    result_df["code5"] = data_df[
        ["RACK_CURRENT_SENSOR_DISCHARGE", "RACK_CURRENT"]
    ].apply(lambda v: protectionmap_5(v[0], v[1]), axis=1)

    result_df["code6"] = data_df["RACK_MAX_CELL_TEMPERATURE"].apply(
        lambda v: protectionmap_6(v)
    )

    result_df["code7"] = data_df["RACK_MIN_CELL_TEMPERATURE"].apply(
        lambda v: protectionmap_7(v)
    )

    result_df["code8"] = data_df[
        ["RACK_MAX_CELL_TEMPERATURE", "RACK_MIN_CELL_TEMPERATURE"]
    ].apply(lambda v: protectionmap_8(v[0], v[1]), axis=1)

    error_code_list = [
        "code1",
        "code2",
        "code3",
        "code4",
        "code5",
        "code6",
        "code7",
        "code8",
    ]

    result_df["OPERRATING_SITE"] = 1

    print(result_df)

    for i in range(len(error_code_list)):

        if result_df[result_df[error_code_list[i]] > 0].empty:
            print("존재안함")
        else:
            print("존재함")

            input_df = result_df[result_df[error_code_list[i]] > 0][
                ["TIMESTAMP", "BANK_ID", "RACK_ID", "OPERATING_SITE"]
            ]
            input_df["ERROR_CODE"] = i + 1

            input_df["LEVEL"] = result_df[error_code_list[i]]

            print(input_df)
            # 존재여부확인해서 저장하는게 필요 존재하면 저장 1도없으면 저장안함

            input_df.to_sql(
                name="protectionmap_feature",
                con=test1.create_engine(),
                schema="public",
                if_exists="append",
                index=false,
            )
