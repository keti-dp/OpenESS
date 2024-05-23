import psycopg2
import argparse
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

def dictfetchall(cursor):
    """
        커서로부터 모든 행을 조회하고, 각 행을 사전(dict) 형태로 반환하는 함수입니다.

        Args:
            cursor (psycopg2.extensions.cursor): 데이터베이스 커서 객체

        Returns:
            list: 각 행을 사전 형태로 저장한 리스트
    """
    columns = [col[0] for col in cursor.description]

    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_ess_db_connection_info(site, host, port, user, pwd):
    
    return {
        "host": host,
        "port": port,
        "dbname":f"ESS_Operating_Site{site}",
        "user": user,
        "password": pwd
    }

def get_dataset(site_id: int, bank_idx: list, rack_idx: list, start_date: str, end_date: str, 
                host: str, port: str, user: str, password: str) -> pd.DataFrame: 
    """
    이 함수는 특정 기간 동안 지정된 사이트, 뱅크, 랙에 대한 데이터를 조회합니다.

    Args:
        site_id (int): 조회하려는 사이트 번호.
        bank_idx (list): 조회하려는 뱅크 ID 리스트.
        rack_idx (list): 조회하려는 랙 ID 리스트.
        start_date (str): 데이터 조회의 시작 시간. (예: '2023-01-01 00:00:00')
        end_date (str): 데이터 조회의 종료 시간.

    Returns:
        pd.DataFrame: 조회된 데이터를 포함하는 데이터프레임을 반환하며, 운영 사이트를 나타내는 추가 열이 포함됩니다.
    """


    # 리스트 또는 튜플을 콤마로 구분된 문자열로 변환
    bank_idx_str = f"(" + ",".join(str(x) for x in bank_idx) + ")"
    rack_idx_str = f"(" + ",".join(str(x) for x in rack_idx) + ")"
    
    conn = psycopg2.connect(**get_ess_db_connection_info(site_id, host, port, user, password))
    
    with open(f"ess_site_{site_id}.sql", "r") as file:
        sql_queries = file.read().format(bank_id=bank_idx_str, rack_id=rack_idx_str, start_date=start_date, end_date=end_date)
    
    with conn.cursor() as cur:
        cur.execute(sql_queries)
        query_dict = dictfetchall(cur)
            
    df = pd.DataFrame(query_dict)
    df = df.sort_values(by=['TIMESTAMP', 'BANK_ID', 'RACK_ID'] ,ascending=True).reset_index(drop=True)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Health indicator(MVF) dataset')
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    parser.add_argument('--site', help="Operating site ID", required=True, type=int)
    parser.add_argument('--bank', help="Number of banks", nargs='+', required=True, type=int)
    parser.add_argument('--rack', help="Number of racks", nargs='+', required=True, type=int)
    parser.add_argument('--start_date', help="Start time, ex)'2022-01-01 00:00:00'", required=True, type=str)
    parser.add_argument('--end_date', help="End time(if not provided, fetches data only at start time)", type=str)
    parser.add_argument('--save_path', help="Save dataset path", type=str, default="/app/data/")

    args = parser.parse_args()

    site_id = args.site
    bank = args.bank
    rack = args.rack
    start_date = args.start_date
    end_date = args.end_date
    save_path = args.save_path

    # TODO: 아래 check 하는 부분을 함수화 하는것이 좋아 보임
    try:
        # 사이트 값의 유효성 검사
        assert site_id in [1, 2, 3, 4], "Invalid site value. Please input a number from 1 to 4."

        # 사이트에 따른 뱅크 및 랙 값의 유효성 검사
        if site_id == 1:
            assert bank == [1], "Invalid bank value for site 1. Only bank 1 is available."
            assert all(1 <= r <= 8 for r in rack), "Invalid rack value. Please input numbers from 1 to 8."
        elif site_id == 2:
            assert all(b in [1, 2] for b in bank), "Invalid bank value for site 2. Only banks 1 and 2 are available."
            assert (1 in bank and all(1 <= r <= 9 for r in rack)) or (2 in bank and all(1 <= r <= 8 for r in rack)), "Invalid rack value. Please input numbers from 1 to 9 for bank 1, 1 to 8 for bank 2."
        elif site_id == 3:
            assert bank == [1], "Invalid bank value for site 3. Only bank 1 is available."
            assert all(1 <= r <= 11 for r in rack), "Invalid rack value. Please input numbers from 1 to 11."
        elif site_id == 4:
            assert bank == [1], "Invalid bank value for site 4. Only bank 1 is available."
            assert all(1 <= r <= 9 for r in rack), "Invalid rack value. Please input numbers from 1 to 9."
        if end_date == None:
            end_date = start_date
    
    except AssertionError as e:
        # 유효성 검사에서 걸린 에러를 출력하고 프로그램 종료
        print(f"Error: {e}")
        exit()

    df = get_dataset(site_id = site_id,
                    bank_idx = bank,
                    rack_idx = rack,
                    start_date = start_date,
                    end_date = end_date,
                    host = host,
                    port = port,
                    user = user,
                    password = password)
    
    df = df.drop_duplicates(subset=['TIMESTAMP', 'BANK_ID', 'RACK_ID'], keep='last').reset_index(drop=True)
    df["SITE_ID"] = site_id
    print(df)

    try:
        # 컨테이너로 만들시 os.getenv(f"DATA_SAVE_PATH") 경로에 데이터 저장
        df.to_csv(save_path + "query_dataset.csv", index=False)

    except OSError as e:
        # 로컬에서 실행시에는 os.getenv(f"DATA_SAVE_PATH")가 없을수 있으니 현재경로에 저장
        df.to_csv("../query_dataset.csv", index=False)
        
    
    
