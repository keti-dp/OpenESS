import pandas as pd
import os
from matplotlib import pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import psycopg2
from datetime import datetime
import math
from pprint import pprint
import sqlalchemy
from sqlalchemy.sql.expression import false
import config
import time 
import logs
import TelegramBot
from pprint import pprint


def dictfetchall(cursor):
    '''
    커서의 모든행을 dict으로 반환하는 함수
        Args:
            cursor (str): a value
            
        Retruns:
            
    '''
    columns = [col[0] for col in cursor.description]
    
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def create_engine():

    test = (
            """postgresql://{_username}:{_password}@{_ip}:{_port}/{_dbname}""".format(
                _username="",
                _password="",
                _ip="",
                _port="",
                _dbname="",
            )
        )
    #print(test)
    engine = sqlalchemy.create_engine(test)
    return engine


def sos_function(safety_inf, inf, value):

    exp1 = 0.25 / math.pow(safety_inf[inf]["upper_safety"] - safety_inf[inf]["maximum_safety"], 2)    
    exp2 = math.pow(value - safety_inf[inf]["maximum_safety"], 2)
    exp3 = 1/(exp1*exp2+1)
    
    return exp3


def calc_sos_safety(data, safety_inf):
    
    f_safety = {"OVER_VOLTAGE":{},
                "UNDER_VOLTAGE":{},
                "VOLTAGE_UNBALANCE":{},
                "OVER_CURRENT":{},
                "OVER_TEMPERATURE":{},
                "UNDER_TEMPERATURE":{},
                "TEMPERATURE_UNBALANCE":{}
               }

    for inf in safety_inf:
        if inf in config.condi_1:     
            for k, v in df_dict[inf].items():
                if df_dict[inf][k] < safety_inf[inf]["maximum_safety"]:
                    f_safety[inf][k] = 1
                else:
                    sos_val = sos_function(safety_inf, inf, v)
                    f_safety[inf][k] = sos_val
    
        elif inf in config.condi_2:
            for k, v in df_dict[inf].items():
                if df_dict[inf][k] > safety_inf[inf]["maximum_safety"]:
                    f_safety[inf][k] = 1
                else:
                    sos_val = sos_function(safety_inf, inf, v)
                    f_safety[inf][k] = sos_val

    sos_dict = {"SOS_SCORE":{}}
    for k in f_safety["OVER_VOLTAGE"].keys():
        sos_score = 1
        for k2 in f_safety.keys():
            f = f_safety[k2][k]
            sos_score = sos_score * f
        sos_dict["SOS_SCORE"][k] = sos_score

    f_safety.update(sos_dict)
    df3 = pd.DataFrame(f_safety)
    df3 = df3.reset_index().rename(columns={"level_0":"BANK_ID", "level_1":"RACK_ID"})
    df3 = pd.concat([df3, df["TIMESTAMP"]], axis=1)

    # OPERATING_SITE = 1 -> 시온유, 2 -> 판리
    df3["OPERATING_SITE"] = 2
    

    return df3


if __name__ == "__main__":
    sosBot = TelegramBot.TelegramBot()
    sosBot.sendMessage(f"판리 시작시간: {datetime.now()}")
    logger = logs.get_logger(lname="Operation2", dirname="./log/", filename="operation2.json")

    while True:
        query_time = (datetime.now() - relativedelta(seconds=5)).strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            #TimescaleDB에서 데이터 획득
            conn = psycopg2.connect(host="",
                               dbname="",
                               user="",
                               password="",
                               port="")

            bank1_query = f"""select * from rack where "TIMESTAMP" between now() - interval '5' second and now() and "BANK_ID" = 1 order by "TIMESTAMP" desc"""
            bank2_query = f"""select * from rack where "TIMESTAMP" between now() - interval '5' second and now() and "BANK_ID" = 2 order by "TIMESTAMP" desc"""

            with conn.cursor() as cur:
                cur.execute(bank1_query)
                bank1_query = dictfetchall(cur)

                cur.execute(bank2_query)
                bank2_query = dictfetchall(cur)
    
            # 데이터 수집 과정에서의 문제로 인한 2번의 쿼리
            df_bank1 = pd.DataFrame(bank1_query, columns=config.col_sel)
            df_bank2 = pd.DataFrame(bank2_query, columns=config.col_sel)
    
            # 중복 데이터 제거 (5초 이내의 데이터중 가장 처음 데이터 획득)
            df_bank1 = df_bank1.dropna().drop_duplicates(["RACK_ID"], keep = 'last')
            df_bank2 = df_bank2.dropna().drop_duplicates(["RACK_ID"], keep = 'last')
    
            df = pd.concat([df_bank1, df_bank2], axis=0).reset_index(drop=True)
   
            # 시간이 다를수 있으므로통일을 위한 작업 
            df["TIMESTAMP"] = query_time
    
            df2 = df.rename(columns={'RACK_MAX_CELL_VOLTAGE':"OVER_VOLTAGE", 
                                     'RACK_MIN_CELL_VOLTAGE':"UNDER_VOLTAGE",
                                     'RACK_CELL_VOLTAGE_GAP': "VOLTAGE_UNBALANCE",
                                     'RACK_CURRENT': "OVER_CURRENT",
                                     'RACK_MAX_CELL_TEMPERATURE': "OVER_TEMPERATURE", 
                                     'RACK_MIN_CELL_TEMPERATURE': "UNDER_TEMPERATURE", 
                                     'RACK_CELL_TEMPERATURE_GAP': "TEMPERATURE_UNBALANCE"}).set_index(["BANK_ID", "RACK_ID"])
                                     
            df_dict = df2.to_dict()
    
            #SOS 계산
            result = calc_sos_safety(df_dict, config.safety_inf)
            #DB에 저장
            engine = create_engine()
            result.to_sql(name="",
                          con=engine,
                          schema="public",
                          if_exists="append",
                          index=false)
            engine.dispose() # DB연결 종료 

            time.sleep(5)

        except Exception as e:
            log_message = f"Error : {e}"
            logger.error(log_message)
            
            sosBot.sendMessage("\U00002757 Panly Error")
            sosBot.sendMessage(df["TIMESTAMP"])
            time.sleep(5)

