# -*- coding:utf-8 -*-

"""
indexing_module.py : timescaleDB index 추가 모듈 코드

Copyright(C) 2023, 윤태일 / KETI / taeil777@keti.re.kr

        ---------------------------------------------------------------------------
        indexing_module.py : timescaleDB index 추가 모듈 코드
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

TimescaleDB에 대한 모듈 코드입니다.

전체적인 코드에 대한 설명은 https://github.com/keti-dp/OpenESS 에서 확인하실 수 있습니다.
"""

import psycopg2
from datetime import date, datetime
from psycopg2.extensions import AsIs
from dateutil.relativedelta import relativedelta
from pytz import timezone
import pprint
import pandas as pd

import argparse


class timescale:

    # 기본 클라이언트 설정
    def __init__(self, ip, port, username, password, dbname):

        try:
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

        except Exception as error:
            print(error)

    def get_table_list(self):

        return self.query("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")

    def get_column_list(self, table_name):

        query_text = f"""SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{table_name}';"""

        return self.query(query_text)

    def add_index(self, table_name, column_name):

        index_name = f"{table_name}_timestamp_{column_name.lower()}_idx"

        query_text = f"""
            CREATE INDEX {index_name} ON public.{table_name} ("TIMESTAMP","{column_name}");
            """

        cursor = self.conn.cursor()
        cursor.execute(query_text)

        # 커밋
        self.conn.commit()

        cursor.execute(f"SELECT * FROM pg_indexes WHERE indexname = '{index_name}'")

        result = cursor.fetchone()

        if result is not None:
            print("인덱스가 성공적으로 생성되었습니다.")
        else:
            print("CREATE INDEX 쿼리 실행에 실패했습니다.")
        cursor.close()

    def drop_index(self, index_name):

        query_text = f"""
        DROP INDEX IF EXISTS public.{index_name};
        """

        cursor = self.conn.cursor()
        cursor.execute(query_text)
        # 커밋
        self.conn.commit()

        for notice in self.conn.notices:
            print(notice)

        cursor.close()

    def query(self, query_text):

        cursor = self.conn.cursor()

        cursor.execute(query_text)

        # 쿼리결과
        result = cursor.fetchall()

        cursor.close()

        df = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)

        return df


if __name__ == "__main__":

    # ArgumentParser 객체 생성
    parser = argparse.ArgumentParser()

    # 각 argument 추가

    parser.add_argument("--host", type=str, default="host", help="PostgreSQL host")
    parser.add_argument("--port", type=int, default=1234, help="PostgreSQL port")
    parser.add_argument("--user", type=str, default="user", help="PostgreSQL user")
    parser.add_argument("--password", type=str, default="password!", help="PostgreSQL password")
    parser.add_argument("--db", type=str, default="dbname", help="PostgreSQL database")

    # 입력받은 argument를 저장할 Namespace 객체 반환
    args = parser.parse_args()

    # 입력받은 argument 출력
    print(args.host)
    print(args.port)
    print(args.user)
    print(args.password)
    print(args.db)

    ip = args.host
    port = args.port
    username = args.user
    password = args.password
    dbname = args.db

    DB = timescale(ip, port, username, password, dbname)
    table_list = DB.get_table_list()
    print(table_list)

    print("------------------------------")
    print("")
    table_name = input("table 명을 입력해주세요 : ")
    print("")

    # 테이블 리스트에서 입력한 테이블이름이 존재하는지 확인
    if table_list.isin([table_name]).any().any():
        print("해당 table이 존재합니다.")
    else:
        print("해당 table이 존재하지 않습니다.")
        exit()

    print("------------------------------")

    column_list = DB.get_column_list(table_name)

    print(column_list)

    print("------------------------------")
    print("")
    column_name = input("column 명을 입력해주세요 : ")
    print("")

    # 컬럼 리스트에서 입력한 컬럼이 존재하는지 확인
    if column_list.isin([column_name]).any().any():
        print("해당 column이 존재합니다.")
    else:
        print("해당 column이 존재하지 않습니다.")
        exit()

    print("------------------------------")

    DB.add_index(table_name, column_name)
