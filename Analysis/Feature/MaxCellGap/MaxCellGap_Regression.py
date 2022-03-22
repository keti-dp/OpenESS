#-*- coding: utf-8 -*-

import sys
from datetime import timedelta, datetime
import pandas as pd
import pandas.io.sql as psql
import psycopg2
import statsmodels.api as sm
import seaborn
from matplotlib import pyplot
from matplotlib.dates import date2num
import matplotlib.dates as mdates
import warnings
import telegram
import os
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    path = os.getcwd()

    """
    1. Telegram Chatbot connection
    """

    # telgm_token =
    # chat_id =
    bot = telegram.Bot(token = telgm_token)
    updates = bot.getUpdates()

    # DB 연결
    # CONNECTION = "postgres://guest_user:####@1.1.1.1:1111/ESS_Operating_Site1"
    with psycopg2.connect(CONNECTION) as conn:
        cursor = conn.cursor()

    # Set Timezone
    query = "SET TIME ZONE 'Asia/Seoul';"
    cursor.execute(query)

    today = datetime.now()

    # 종료시간
    e_time = today.strftime('%Y-%m-%d 00:00:00')

    # 시작시간
    s_time = (today - timedelta(days= 60)).strftime('%Y-%m-%d 00:00:00')

    # 기본 쿼리
    query = """
    SELECT date_trunc('day', r."TIMESTAMP") as "DAY",
        "RACK_ID",
        MAX("RACK_CELL_VOLTAGE_GAP") as MAX_RACK_CELL_VOLTAGE_GAP
    FROM rack r 
    WHERE r."TIMESTAMP" >= '{stime}'
        and r."TIMESTAMP" < '{etime}'
    group by "DAY", "RACK_ID";
    """.format(stime= s_time, etime=e_time)

    df_rack = psql.read_sql(query, conn)
    dict_rack = df_rack.to_dict()
    cursor.close()

    # 랙별로 결과 도출하기
    for i in range(1,9):
        df_rack_temp = df_rack[df_rack['RACK_ID']==i]
        df_rack_temp = df_rack_temp.reset_index(drop=True)
        df_rack_temp = df_rack_temp.reset_index()
        date_list = []
        for j in range(len(df_rack_temp)):
            date = df_rack_temp['DAY'][j] + timedelta(days=1)
            date_list.append(date.date())
        df_rack_temp['date'] = date_list
        df_rack_temp['date'] = df_rack_temp['date'].apply(date2num)

        reg_temp = sm.OLS.from_formula("max_rack_cell_voltage_gap ~ index", df_rack_temp).fit()

        index_coeff = pd.read_html(reg_temp.summary().tables[1].as_html())[0][1][2]
        if float(index_coeff) >= 0.0004:
            message = """
            !!WARNING!!\n[RACK {}] 최근 2개월동안 셀 전압차이 최대값이 증가하고 있습니다.\n일별 증가량: {}v
            """.format(i, index_coeff)
            print('Warning')
            print('[RACK {}] 최근 2개월동안 셀 전압차이 최대값이 증가하고 있습니다.'.format(i))
            print('일별 증가량: {}v'.format(index_coeff))
            bot.sendMessage(chat_id = chat_id, text = message)

            # 그래프 시각화
            seaborn.set(style = 'whitegrid')
            fig, ax = pyplot.subplots(figsize=(16,8))
            pyplot.title('RACK{}'.format(i))
            seaborn.regplot('date', 'max_rack_cell_voltage_gap', data=df_rack_temp, ax=ax, scatter_kws={"color": "black"}, line_kws={"color": "red"})
            myFmt = mdates.DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_formatter(myFmt)
            ax.tick_params(labelrotation=45)
            pyplot.savefig(path+'/warning_pic.png')

            bot.send_photo(chat_id=chat_id, photo=open('warning_pic.png', 'rb'))
