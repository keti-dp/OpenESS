#-*- coding: utf-8 -*-

import psycopg2
import pandas.io.sql as psql
import yaml
import os
currentpath = os.getcwd()

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
               "    WHERE (\"TIMESTAMP\" > '2021-10-01 00:00:00' and \"TIMESTAMP\" < '2021-11-01 00:00:00')"

        with psycopg2.connect(self.CONNECTION) as self.conn:
            df = psql.read_sql(query, self.conn)

        return df


def write_outlier(df):
    """
    yaml 파일 읽어서 딕셔너리 형태로 반환
    """

    mean = dict(df.mean())
    std = dict(df.std())

    with open(currentpath+'/lib/outlier_3sigma_mean.yaml', 'w') as f:
        for key, val in mean.items():
            keyval = "{}: {}\n".format(key, val)
            f.write(keyval)

    with open(currentpath+'/lib/outlier_3sigma_std.yaml', 'w') as f:
        for key, val in std.items():
            keyval = "{}: {}\n".format(key, val)
            f.write(keyval)

if __name__ == "__main__":
    setting = read_setting()

    """DB Instance 생성"""
    Outlier_DB = ESSDatabase(dbname=setting['dbname'],
                     user=setting['user'],
                     password=setting['password'])
    """데이터 로드"""
    outlier_dataset = Outlier_DB.query_select()
    print('dataset load')

    """3시그마 규칙"""
    write_outlier(outlier_dataset)
    print('mean, sigma save')