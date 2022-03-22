#-*- coding: utf-8 -*-
"""

    Title: DataLoad_DB

"""

import sys
import psycopg2
import pandas as pd
from datetime import datetime
import yaml
from pprint import pprint
import pandas.io.sql as psql


class ESSDatabase:
    """ESS_"""

    # 기본 클라이언트 설정
    def __init__(self, dbname, user, password):

        # timescale DB 연결
        self.CONNECTION = (
            """postgres://{_user}:{_password}@{_ip}:{_port}/{_dbname}""".format(
                _ip='#.#.#.#', _port='####', _dbname=dbname, _user=user, _password=password
            )
        )

    def query_select(self):
        """
        query 실행

        :return: pandas dataframe
        """

        query= "SELECT * FROM bank " \
               "    WHERE (\"TIMESTAMP\" > '2021-10-01 00:00:00' and \"TIMESTAMP\" < '2021-10-01 00:10:00')"

        with psycopg2.connect(self.CONNECTION) as self.conn:
            df = psql.read_sql(query, self.conn)

        return df

if __name__ == "__main__":

    """DB Instance 생성"""
    DB = ESSDatabase(dbname=setting['dbname'],
                     user=setting['user'],
                     password=setting['password'])

    dataset = DB.query_select()
    print(dataset.head(10), '\n')
    print(' ', len(dataset),'rows & ', len(dataset.columns),'cols loaded')