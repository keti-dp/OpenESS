# -*- coding:utf-8 -*-

"""
db_interpolation.py : 두 곳에 존재하는 데이터베이스에 하루치 데이터에서 없는 부분을 서로 보간해주는 모듈 코드

Copyright(C) 2023, 윤태일 / KETI / taeil777@keti.re.kr

        ---------------------------------------------------------------------------
        db_interpolation.py : 두 곳에 존재하는 데이터베이스에 하루치 데이터에서 없는 부분을 서로 보간해주는 모듈 코드
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
import pandas as pd
from psycopg2 import sql
from psycopg2.extras import Json
from datetime import datetime, timedelta
import time
from pprint import pprint

# from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine
from sqlalchemy.sql.expression import false
import traceback


def get_data_in_timescale(oper, mode, bank_id, rack_id=None):
    """타임스케일디비에서 데이터 가져오는 코드"""

    # TimescaleDB 연결 정보
    db_host = ""
    db_port = ""
    db_name = oper
    db_user = ""
    db_password = ""

    db2_host = ""
    db2_port = ""
    db2_name = oper
    db2_user = ""
    db2_password = ""

    gcp_timescaledb_connection = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
    )

    local_timescaledb_connection = psycopg2.connect(
        host=db2_host,
        port=db2_port,
        dbname=db2_name,
        user=db2_user,
        password=db2_password,
    )

    current_time = (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    # 1일 전의 시간 계산
    one_day_ago = (
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    ).strftime("%Y-%m-%d %H:%M:%S")

    print(one_day_ago)
    print(current_time)

    query = ""
    add_query = ""

    if mode == "rack":
        add_query = f"""AND "RACK_ID" = {rack_id} """

    query = f"""
    SET TIME ZONE 'Asia/Seoul';
    SELECT * FROM {mode} WHERE "BANK_ID" = {bank_id} {add_query}AND "TIMESTAMP" BETWEEN '{one_day_ago} Asia/Seoul' AND '{current_time} Asia/Seoul' ORDER BY "TIMESTAMP" DESC;"""

    # 1 뱅크기준 쿼리문
    # 2 rack 기준 쿼리문

    gcp_df = pd.read_sql(query, gcp_timescaledb_connection)

    local_df = pd.read_sql(query, local_timescaledb_connection)

    # psycopg2는 UTC로 밖에 못가져온다네??

    # 타임존 변환
    gcp_df["TIMESTAMP"] = gcp_df["TIMESTAMP"].dt.tz_convert("Asia/Seoul")
    local_df["TIMESTAMP"] = local_df["TIMESTAMP"].dt.tz_convert("Asia/Seoul")

    save_to_local_df = gcp_df[
        ~gcp_df["TIMESTAMP"].isin(local_df["TIMESTAMP"])
    ]  # << local_df에 없는 gcp_df데이터들 , local_df에 저장할 데이터들
    save_to_gcp_df = local_df[
        ~local_df["TIMESTAMP"].isin(gcp_df["TIMESTAMP"])
    ]  # << gcp_df에 없는 local_df데이터들, gcp_df에 저장할 데이터들

    # A 데이터프레임에 있는 TIMESTAMP가 B 데이터프레임에 없을 경우 해당 행들 추출

    print(save_to_local_df)

    print(save_to_gcp_df)

    # with 문을 사용하여 커서 생성
    with local_timescaledb_connection.cursor() as cur:
        # 데이터프레임의 각 행을 순회
        for i, row in save_to_local_df.iterrows():
            # 행을 딕셔너리로 변환
            row_dict = row.to_dict()

            # JSON 형식의 컬럼이 있다면, psycopg2.extras.Json을 사용하여 변환
            for key, value in row_dict.items():
                if isinstance(value, dict):
                    row_dict[key] = Json(value)

            # 쿼리 생성
            insert = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                mode,
                sql.SQL(",").join(map(sql.Identifier, row_dict.keys())),
                sql.SQL(",").join(map(sql.Placeholder, row_dict.keys())),
            )

            # 쿼리 실행
            cur.execute(insert, row_dict)

        # 변경사항 커밋
        local_timescaledb_connection.commit()

    with gcp_timescaledb_connection.cursor() as cur:
        # 데이터프레임의 각 행을 순회
        for i, row in save_to_gcp_df.iterrows():
            # 행을 딕셔너리로 변환
            row_dict = row.to_dict()

            # JSON 형식의 컬럼이 있다면, psycopg2.extras.Json을 사용하여 변환
            for key, value in row_dict.items():
                if isinstance(value, dict):
                    row_dict[key] = Json(value)

            # 쿼리 생성
            insert = sql.SQL("INSERT INTO bank ({}) VALUES ({})").format(
                sql.SQL(",").join(map(sql.Identifier, row_dict.keys())),
                sql.SQL(",").join(map(sql.Placeholder, row_dict.keys())),
            )

            # 쿼리 실행
            cur.execute(insert, row_dict)

        # 변경사항 커밋
        gcp_timescaledb_connection.commit()


def main():
    OPERATION_SITE_DICT = {
        # "ESS_Operating_Site1": {1: 8},
        "ESS_Operating_Site2": {1: 9, 2: 8},
        "ESS_Operating_Site3": {1: 11},
        "ESS_Operating_Site4": {1: 9},
    }

    for site, info in OPERATION_SITE_DICT.items():
        print("site : ", site)
        num_bank = info.keys()  # 뱅크개수 리스트로 반환

        for bank_id in num_bank:
            # 여기선 뱅크를 처리
            try:
                print("bank_id : ", bank_id)
                get_data_in_timescale(site, "bank", bank_id)
            except Exception as e:
                traceback.print_exc()
                continue  # 에러나면 다음걸로
            num_rack = info[bank_id]  # 뱅크에 따른 랙 개수
            for rack_id in range(1, num_rack + 1):
                try:
                    print("rack_id : ", rack_id)
                    get_data_in_timescale(site, "rack", bank_id, rack_id)
                except Exception as e:
                    traceback.print_exc()
                    continue  # 에러나면 다음걸로


# 일단 처음에 데이터를 다 넣어야함일단
if __name__ == "__main__":
    # 크론탭으로 동작하게하기

    main()
