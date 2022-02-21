# -*- coding:utf-8 -*-

"""
timescale_input_data.py : timescaleDB 모듈 코드

Copyright(C) 2022, 윤태일 / KETI / taeil777@keti.re.kr

        ---------------------------------------------------------------------------
        timescale_input_data.py : Timescale DB에 데이터를 입력하는 모듈 코드
        Copyright(C) 2021, 윤태일 / KETI / taeil777@keti.re.kr

        이 프로그램은 자유 소프트웨어입니다. 당신은 자유 소프트웨어 재단이 공표한 GNU 일반 공중 라이선스 버전 2 또는 
        그 이후 버전을 임의로 선택해서 그 규정에 따라 프로그램을 수정하거나 재배포할 수 있습니다.

        이 프로그램은 유용하게 사용될 수 있을 것이라는 희망에서 배포되고 있지만 어떠한 형태의 보증도 제공하지 않습니다. 
        상품성 또는 특정 목적 적합성에 대한 묵시적 보증 역시 제공하지 않습니다. 보다 자세한 내용은 GNU 일반 공중 라이선스를 참고하시기 바랍니다.

        GNU 일반 공중 라이선스는 이 프로그램과 함께 제공됩니다. 만약, 라이선스를 받지 못했다면, 
        자유 소프트웨어 재단으로 문의하기 바랍니다. 
        주소: Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
        ---------------------------------------------------------------------------


최신 테스트 버전 : 1.0.0 ver
최신 안정화 버전 : 1.0.0 ver

TimescaleDB에 대한 모듈 코드입니다. 운영사이트마다 구성되어있는 ESS 구조가 다르기때문에 세부 코드가 다릅니다.

다음 세 가지 기능을 가지고 있습니다.

1. hypertable create

2. data input 

3. data query 

전체적인 코드에 대한 설명은 https://github.com/keti-dp/OpenESS 에서 확인하실 수 있습니다.
"""

import psycopg2
from datetime import date, datetime
from psycopg2.extensions import AsIs
from dateutil.relativedelta import relativedelta
from pytz import timezone
import pprint


class timescale:

    # 기본 클라이언트 설정
    def __init__(self):
        # timescale DB 연결
        self.CONNECTION = """DB접속명령어"""
        with psycopg2.connect(self.CONNECTION) as self.conn:
            self.cursor = self.conn.cursor()

        print("---------timescaledb connected----------")

    # 테이블 생성
    def create_hypertable(self, table_name):

        # 하이퍼 테이블 생성 SQL 문
        # Not Null 은 문자그대로 비어있으면 안됨

        if "bank" in table_name:

            # Bank 생성 SQL문
            query_create_sensordata_table = """CREATE TABLE IF NOT EXISTS "{_tablename}" (
                                                "TIMESTAMP" timestamptz NOT NULL,
                                                "BANK_ID" int4 NOT NULL,
                                                "BANK_SOC" float8 NULL,
                                                "BANK_SOH" float8 NULL,
                                                "BANK_DC_VOLT" float8 NULL,
                                                "BANK_DC_CURRENT" float8 NULL,
                                                "BANK_POWER" float8 NULL,
                                                "MAX_CELL_VOLTAGE_OF_BANK" float8 NULL,
                                                "MIN_CELL_VOLTAGE_OF_BANK" float8 NULL,
                                                "MAX_CELL_TEMPERATURE_OF_BANK" float8 NULL,
                                                "MIN_CELL_TEMPERATURE_OF_BANK" float8 NULL,
                                                "MAX_MODULE_TEMPERATURE" float8 NULL, 
                                                "MIN_MODULE_TEMPERATURE" float8 NULL,
                                                "MAX_MODULE_HUMIDITY" float8 NULL, 
                                                "MIN_MODULE_HUMIDITY" float8 NULL,
                                                "RACK_TEMPERATURE_IMBALANCE_WARNING" int4 NULL,
                                                "RACK_UNDER_TEMPERATURE_WARNING" int4 NULL,
                                                "RACK_OVER_TEMPERATURE_WARNING" int4 NULL,
                                                "RACK_VOLTAGE_IMBALANCE_WARNING" int4 NULL,
                                                "RACK_UNDER_VOLTAGE_PROTECTION_WARNING" int4 NULL,
                                                "RACK_OVER_VOLTAGE_PROTECTION_WARNING" int4 NULL,
                                                "RACK_OVER_CURRENT_CHARGE_WARNING" int4 NULL,
                                                "RACK_OVER_CURRENT_DISCHARGE_WARNING" int4 NULL,
                                                "RACK_TRAY_VOLTAGE_IMBALANCE_WARNING" int4 NULL,
                                                "RACK_TEMPERATURE_IMBALANCE_FAULT" int4 NULL,
                                                "RACK_UNDER_TEMPERATURE_FAULT" int4 NULL,
                                                "RACK_OVER_TEMPERATURE_FAULT" int4 NULL,
                                                "RACK_VOLTAGE_IMBALANCE_FAULT" int4 NULL,
                                                "RACK_UNDER_VOLTAGE_PROTECTION_FAULT" int4 NULL,
                                                "RACK_OVER_VOLTAGE_PROTECTION_FAULT" int4 NULL,
                                                "RACK_OVER_CURRENT_CHARGE_FAULT" int4 NULL,
                                                "RACK_OVER_CURRENT_DISCHARGE_FAULT" int4 NULL,
                                                "RACK_CHARGE_RELAY_PLUS_FAULT_STATUS" int4 NULL,
                                                "RACK_DISCHARGE_RELAY_MINUS_FAULT_STATUS" int4 NULL,
                                                "RACK_FUSE_MINUS_FAULT_STATUS" int4 NULL,
                                                "RACK_FUSE_PLUS_FAULT_STATUS" int4 NULL,
                                                "RACK_TRAY_RACK_COMMUNICATION_FAULT" int4 NULL,
                                                "MASTER_RACK_COMMUNICATION_FAULT" json NULL,                                              
                                                "BATTERY_STATUS_FOR_STANDBY" int4 NULL,
                                                "BATTERY_STATUS_FOR_CHARGE" int4 NULL,
                                                "BATTERY_STATUS_FOR_DISCHARGE" int4 NULL,
                                                "BATTERY_STATUS_FOR_FAULT" int4 NULL,
                                                "BATTERY_STATUS_FOR_WARNING" int4 NULL,
                                                "RACK_TO_MASTER_BMS_COMMUNICATION_STATUS" int4 NULL,
                                                "CHARGING_STOP_OF_STATUS" int4 NULL,
                                                "DISCHARGING_STOP_OF_STATUS" int4 NULL,
                                                "EMERGENCY_STATUS" int4 NULL,
                                                "TVOC_STATUS" int4 NULL,
                                                "BATTERY_STATUS_FOR_RUN" int4 NULL,
                                                "ORDER_SOURCE" int4 NULL,
                                                "EMERGENCY_SOURCE" int4 NULL
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

        if "rack" in table_name:
            # Rack 생성 SQL문
            query_create_sensordata_table = """CREATE TABLE IF NOT EXISTS "{_tablename}" (
                                            "TIMESTAMP" timestamptz NOT NULL,
                                            "BANK_ID" int4 NOT NULL,
                                            "RACK_ID" int4 NOT NULL,
                                            "RACK_SOC" float8 NULL,
                                            "RACK_SOH" float8 NULL,
                                            "RACK_VOLTAGE" float8 NULL,
                                            "RACK_CURRENT" float8 NULL,                   
                                            "RACK_MAX_CELL_VOLTAGE" float8 NULL,
                                            "RACK_MAX_CELL_VOLTAGE_POSITION" float8 NULL,
                                            "RACK_MIN_CELL_VOLTAGE" float8 NULL,
                                            "RACK_MIN_CELL_VOLTAGE_POSITION" float8 NULL,
                                            "RACK_CELL_VOLTAGE_GAP" float8 NULL,
                                            "RACK_CELL_VOLTAGE_AVERAGE" float8 NULL,
                                            "RACK_MAX_CELL_TEMPERATURE" float8 NULL,
                                            "RACK_MAX_CELL_TEMPERATURE_POSITION" float8 NULL,
                                            "RACK_MIN_CELL_TEMPERATURE" float8 NULL,
                                            "RACK_MIN_CELL_TEMPERATURE_POSITION" float8 NULL,
                                            "RACK_CELL_TEMPERATURE_GAP" float8 NULL,   
                                            "RACK_MAX_MODULE_TEMPERATURE" float8 NULL, 
                                            "RACK_MAX_MODULE_TEMPERATURE_POSITION" float8 NULL, 
                                            "RACK_MIN_MODULE_TEMPERATURE" float8 NULL, 
                                            "RACK_MIN_MODULE_TEMPERATURE_POSITION" float8 NULL, 
                                            "RACK_MAX_MODULE_HUMIDITY" float8 NULL, 
                                            "RACK_MAX_MODULE_HUMIDITY_POSITION" float8 NULL, 
                                            "RACK_MIN_MODULE_HUMIDITY" float8 NULL, 
                                            "RACK_MIN_MODULE_HUMIDITY_POSITION" float8 NULL, 
                                            "RACK_DISCHARGE_RELAY_MINUS_STATUS" int4 NULL,
                                            "RACK_CHARGE_RELAY_PLUS_STATUS" int4 NULL,
                                            "RACK_CELL_BALANCE_STATUS" int4 NULL,
                                            "RACK_CURRENT_SENSOR_DISCHARGE" int4 NULL,
                                            "RACK_CURRENT_SENSOR_CHARGE" int4 NULL,
                                            "RACK_STATUS_FOR_RUN" int4 NULL,
                                            "RACK_STATUS_FOR_FAULT" int4 NULL,
                                            "RACK_STATUS_FOR_WARNING" int4 NULL,
                                            "RACK_STATUS_FOR_Online" int4 NULL,
                                            "RACK_TEMPERATURE_IMBALANCE_WARNING" int4 NULL,
                                            "RACK_UNDER_TEMPERATURE_WARNING" int4 NULL,
                                            "RACK_OVER_TEMPERATURE_WARNING" int4 NULL,
                                            "RACK_VOLTAGE_IMBALANCE_WARNING" int4 NULL,
                                            "RACK_UNDER_VOLTAGE_PROTECTION_WARNING" int4 NULL,
                                            "RACK_OVER_VOLTAGE_PROTECTION_WARNING" int4 NULL,
                                            "RACK_OVER_CURRENT_CHARGE_WARNING" int4 NULL,
                                            "RACK_OVER_CURRENT_DISCHARGE_WARNING" int4 NULL,
                                            "RACK_TRAY_VOLATGE_IMBALANCE_WARNING" int4 NULL,
                                            "RACK_TEMPERATURE_IMBALANCE_FAULT" int4 NULL,
                                            "RACK_UNDER_TEMPERATURE_FAULT" int4 NULL,
                                            "RACK_OVER_TEMPERATURE_FAULT" int4 NULL,
                                            "RACK_VOLTAGE_IMBALANCE_FAULT" int4 NULL,
                                            "RACK_UNDER_VOLTAGE_PROTECTION_FAULT" int4 NULL,
                                            "RACK_OVER_VOLTAGE_PROTECTION_FAULT" int4 NULL,
                                            "RACK_OVER_CURRENT_CHARGE_FAULT" int4 NULL,
                                            "RACK_OVER_CURRENT_DISCHARGE_FAULT" int4 NULL,
                                            "RACK_CHARGE_RELAY_PLUS_FAULT_STATUS" int4 NULL,
                                            "RACK_DISCHARGE_RELAY_MINUS_FAULT_STATUS" int4 NULL,
                                            "RACK_FUSE_MINUS_FAULT_STATUS" int4 NULL,
                                            "RACK_FUSE_PLUS_FAULT_STATUS" int4 NULL,
                                            "RACK_TRAY_RACK_COMMUNICATION_FAULT" int4 NULL,
                                            "RACK_MODULE_FAULT" json NULL
                                            );""".format(
                _tablename=table_name
            )
            #   FOREIGN KEY (Bank_ID) REFERENCES BANK (Bank_ID)

            query_create_sensordata_hypertable = """SELECT create_hypertable('{_tablename}', 'TIMESTAMP', if_not_exists => TRUE);""".format(
                _tablename=table_name
            )

            self.cursor.execute(query_create_sensordata_table)
            self.cursor.execute(query_create_sensordata_hypertable)
            # commit changes to the database to make changes persistent
            self.conn.commit()
            self.cursor.close()

        if "pcs" in table_name:

            # PCS 생성 SQL문
            query_create_sensordata_table = """CREATE TABLE IF NOT EXISTS "{_tablename}" (
                                            "TIMESTAMP" timestamptz NOT NULL,

                                            "AI_VDC" float8 NULL,
                                            "AI_IDC" float8 NULL,
                                            "AI_PDC" float8 NULL,
                                            "AI_FREQ" float8 NULL,
                                            "AI_VAB_RMS" float8 NULL,
                                            "AI_VBC_RMS" float8 NULL,
                                            "AI_VCA_RMS" float8 NULL,
                                            "AI_IAS_RMS" float8 NULL,
                                            "AI_IBS_RMS" float8 NULL,
                                            "AI_ICS_RMS" float8 NULL,
                                            "AI_SAC" float8 NULL,
                                            "AI_PAC" float8 NULL,
                                            "AI_QAC" float8 NULL,
                                            "AI_PF" float8 NULL,
                                            "AI_C_KWH_ACH" float8 NULL,
                                            "AI_C_KWH_ACL" float8 NULL,
                                            "AI_D_KWH_ACH" float8 NULL,
                                            "AI_D_KWH_ACL" float8 NULL,
                                            "AI_C_KWH_DCH" float8 NULL,
                                            "AI_C_KWH_DCL" float8 NULL,
                                            "AI_D_KWH_DCH" float8 NULL,
                                            "AI_D_KWH_DCL" float8 NULL,
                                            "CMD_KW" float8 NULL,
                                            "CMD_KVAR" float8 NULL,
                                            "CMD_VDC_REF" float8 NULL,
                                            "ST_RUN" float8 NULL,
                                            "ST_STOP" float8 NULL,
                                            "ST_READY" float8 NULL,
                                            "ST_MODE_L_R " float8 NULL,
                                            "ST_MODE_CV " float8 NULL,
                                            "ST_FAULT" float8 NULL,
                                            "ST_CHARGE" float8 NULL,
                                            "ST_DISCHARGE" float8 NULL,
                                            "DI_DK" float8 NULL,
                                            "DI_AK" float8 NULL,
                                            "DI_CK" float8 NULL,
                                            "DI_TEMP" float8 NULL,
                                            "DI_SPD" float8 NULL,
                                            "DI_DS" float8 NULL,
                                            "DI_START" float8 NULL,
                                            "DI_ES" float8 NULL,
                                            "FLT_OVAR" float8 NULL,
                                            "FLT_UVAR" float8 NULL,
                                            "FLT_OFR" float8 NULL,
                                            "FLT_UFR" float8 NULL,
                                            "FLT_OCAR" float8 NULL,
                                            "FLT_OVDR" float8 NULL,
                                            "FLT_UVDR" float8 NULL,
                                            "FLT_OCDR" float8 NULL,
                                            "FLT_CFD" float8 NULL,
                                            "FLT_OTR" float8 NULL,
                                            "FLT_SPD" float8 NULL,
                                            "FLT_RVET" float8 NULL
                                        );""".format(
                _tablename=table_name
            )

            #    FOREIGN KEY (Bank_ID) REFERENCES RACK (Bank_ID)

            query_create_sensordata_hypertable = """SELECT create_hypertable('{_tablename}', 'TIMESTAMP', if_not_exists => TRUE);""".format(
                _tablename=table_name
            )

            self.cursor.execute(query_create_sensordata_table)
            self.cursor.execute(query_create_sensordata_hypertable)
            # commit changes to the database to make changes persistent
            self.conn.commit()
            self.cursor.close()

        if "etc" in table_name:

            # 생성 ETC문
            query_create_sensordata_table = """CREATE TABLE IF NOT EXISTS "{_tablename}" (
                                            "TIMESTAMP" timestamptz NOT NULL,
                                            "SENSOR1_TEMPERATURE" float8 NULL,
                                            "SENSOR1_HUMIDITY" float8 NULL,
                                            "SENSOR2_TEMPERATURE" float8 NULL,
                                            "SENSOR2_HUMIDITY" float8 NULL,

                                            "ACTIVE_POWER_TOTAL" float8 NULL,
                                            "ACTIVE_ENERGY_TOTAL_HIGH" float8 NULL
                                            );""".format(
                _tablename=table_name
            )

            #    FOREIGN KEY (Bank_ID) REFERENCES RACK (Bank_ID)

            query_create_sensordata_hypertable = """SELECT create_hypertable('{_tablename}', 'TIMESTAMP', if_not_exists => TRUE);""".format(
                _tablename=table_name
            )

            self.cursor.execute(query_create_sensordata_table)
            self.cursor.execute(query_create_sensordata_hypertable)
            # commit changes to the database to make changes persistent
            self.conn.commit()
            self.cursor.close()

        print("---- Table Created ----")

    # 데이터 인풋
    def Bank_input_data(self, datetime, input_list, operation_site):

        # input_logger = logs.get_logger("log3", "./Data_Ingestion/log/", "input.logs")
        input_logger = logs.get_logger(operation_site, "/home/keti_iisrc/test/log/")

        # year = "{:%y}".format(datetime)
        # month = "{:%m}".format(datetime)
        # day = "{:%d}".format(datetime)

        # 매월 1일이면 테이블 생성
        # table_name = """{_year}{_month}_bank""".format(_year=year, _month=month)

        table_name = "bank"

        # if day == "1":
        #     self.create_hypertable(table_name)

        # Bank data input
        SQL = """INSERT INTO "{_tablename}" VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s);""".format(
            _tablename=table_name
        )
        # sensors = [('a', 'floor'), ('a', 'ceiling'), ('b', 'floor'), ('b', 'ceiling')]

        # 시간추가

        input_list.insert(0, datetime)

        # 리스트를 튜플로 변환

        Bank_data = tuple(input_list)

        print(Bank_data)
        print(len(Bank_data))

        cursor = self.conn.cursor()

        try:
            cursor.execute(SQL, Bank_data)
        except (Exception, psycopg2.Error) as error:
            print(error.pgerror)
            input_logger.error("Bank data input error : ", error.pgerror)
        self.conn.commit()
        cursor.close()
        print("---- Bank data input success ----")
        # input_logger.info("Bank data input success")

    def Rack_input_data(self, datetime, input_list, operation_site):
        # Rack data input

        # input_logger = logs.get_logger("log3", "./Data_Ingestion/log/", "input.logs")
        input_logger = logs.get_logger(operation_site, "/home/keti_iisrc/test/log/")

        # year = "{:%y}".format(datetime)
        # month = "{:%m}".format(datetime)
        # day = "{:%d}".format(datetime)

        # 매월 1일이면 테이블 생성
        # table_name = """{_year}{_month}_rack""".format(_year=year, _month=month)
        table_name = "rack"

        # if day == "1":
        #     self.create_hypertable(table_name)

        SQL = """INSERT INTO "{_tablename}" VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s, %s);""".format(
            _tablename=table_name
        )
        # sensors = [('a', 'floor'), ('a', 'ceiling'), ('b', 'floor'), ('b', 'ceiling')]
        Rack_data = input_list

        cursor = self.conn.cursor()

        # UTC = datetime.now(timezone("UTC"))

        for Rack in Rack_data:
            try:
                Rack.insert(0, datetime)

                data = tuple(Rack)
                cursor.execute(SQL, data)

                print("---- Rack data input success ----")
                # input_logger.info("Rack data input success")
            except (Exception, psycopg2.Error) as error:
                print(error.pgerror)
                input_logger.error("Rack data input error : ", error.pgerror)
        self.conn.commit()
        cursor.close()

    def PCS_input_data(self, datetime, input_list, operation_site):
        # PCS data input

        # input_logger = logs.get_logger("log3", "./Data_Ingestion/log/", "input.logs")
        input_logger = logs.get_logger(operation_site, "/home/keti_iisrc/test/log/")

        # year = "{:%y}".format(datetime)
        # month = "{:%m}".format(datetime)
        # day = "{:%d}".format(datetime)

        # 매월 1일이면 테이블 생성

        # table_name = """{_year}{_month}_pcs""".format(_year=year, _month=month)

        table_name = "pcs"

        # if day == "1":
        #     self.create_hypertable(table_name)

        SQL = """INSERT INTO "{_tablename}" VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s);""".format(
            _tablename=table_name
        )
        # sensors = [('a', 'floor'), ('a', 'ceiling'), ('b', 'floor'), ('b', 'ceiling')]

        # 시간추가
        # UTC = datetime.now(timezone("UTC"))
        input_list.insert(0, datetime)

        # 리스트를 튜플로 변환
        PCS_data = tuple(input_list)
        print(len(PCS_data))

        cursor = self.conn.cursor()

        try:

            cursor.execute(SQL, PCS_data)
            print("---- PCS data input success ----")
            # input_logger.info("PCS data input success")
        except (Exception, psycopg2.Error) as error:
            print(error.pgerror)
            # input_logger.error("PCS data input error : ", error.pgerror)
        self.conn.commit()
        cursor.close()

    def ETC_input_data(self, datetime, input_list, operation_site):
        # PCS data input

        # input_logger = logs.get_logger("log3", "./Data_Ingestion/log/", "input.logs")
        input_logger = logs.get_logger(operation_site, "/home/keti_iisrc/test/log/")

        # year = "{:%y}".format(datetime)
        # month = "{:%m}".format(datetime)
        # day = "{:%d}".format(datetime)

        # 매월 1일이면 테이블 생성
        # table_name = """{_year}{_month}_etc""".format(_year=year, _month=month)

        table_name = "etc"

        # if day == "1":
        #     self.create_hypertable(table_name)

        SQL = """INSERT INTO "{_tablename}" VALUES (%s, %s, %s, %s, %s, %s, %s);""".format(
            _tablename=table_name
        )
        # sensors = [('a', 'floor'), ('a', 'ceiling'), ('b', 'floor'), ('b', 'ceiling')]

        # 시간추가
        # UTC = datetime.now(timezone("UTC"))
        input_list.insert(0, datetime)

        # 리스트를 튜플로 변환
        ETC_data = tuple(input_list)

        cursor = self.conn.cursor()

        try:

            cursor.execute(SQL, ETC_data)
            print("---- ETC data input success ----")
            # input_logger.info("ETC data input success")
        except (Exception, psycopg2.Error) as error:
            print(error.pgerror)
            # input_logger.error("ETC data input error : ", error.pgerror)
        self.conn.commit()
        cursor.close()

    def testdata_input_data(self, starttime, endtime):

        # PCS data input

        SQL = "INSERT INTO test VALUES (%s, %s, %s, %s, %s);"
        # sensors = [('a', 'floor'), ('a', 'ceiling'), ('b', 'floor'), ('b', 'ceiling')]

        # 시간추가
        UTC = datetime.now(timezone("UTC"))

        # 리스트를 튜플로 변환
        test_data = (UTC, UTC, UTC, UTC, UTC)
        cursor = self.conn.cursor()

        try:
            cursor.execute(SQL, test_data)
            print("---- test data input success ----")
        except (Exception, psycopg2.Error) as error:
            print(error.pgerror)
        self.conn.commit()
        cursor.close()

    def query_data(self, query_text):

        cursor = self.conn.cursor()
        query = query_text
        # query = """SELECT "TIMESTAMP" FROM public.bank;"""
        cursor.execute(query)

        s1 = cursor.fetchall()

        cursor.close()

        return s1


if __name__ == "__main__":

    test = timescale()
