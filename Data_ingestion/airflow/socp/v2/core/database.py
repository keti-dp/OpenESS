"""
데이터베이스 관리 모듈

PostgreSQL 데이터베이스 연결 및 데이터 처리 기능을 제공합니다.
"""

import pandas as pd
from typing import List, Tuple, Optional
from airflow.providers.postgres.hooks.postgres import PostgresHook


class DatabaseManager:
    """
    데이터베이스 연결 및 데이터 처리를 위한 관리 클래스
    """

    def __init__(self, conn_id: str):
        """
        초기화

        Args:
            conn_id: Airflow 데이터베이스 연결 ID
        """
        self.conn_id = conn_id
        self.hook = PostgresHook(postgres_conn_id=conn_id)

    def execute_query_to_dataframe(self, query: str) -> pd.DataFrame:
        """
        쿼리 실행 결과를 DataFrame으로 반환

        Args:
            query: SQL 쿼리 문자열

        Returns:
            쿼리 결과를 담은 DataFrame
        """
        return self.hook.get_pandas_df(sql=query)

    def execute_query_with_columns(self, query: str) -> Tuple[List[Tuple], List[str]]:
        """
        쿼리 실행 결과와 컬럼 정보를 함께 반환

        Args:
            query: SQL 쿼리 문자열

        Returns:
            (결과 튜플 리스트, 컬럼 이름 리스트)
        """
        conn = self.hook.get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute(query)
            result = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            return result, column_names
        finally:
            cursor.close()
            conn.close()

    def insert_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        """
        DataFrame 데이터를 데이터베이스 테이블에 삽입

        Args:
            df: 삽입할 데이터가 담긴 DataFrame
            table_name: 대상 테이블 이름

        Raises:
            Exception: 데이터 삽입 중 오류 발생 시
        """
        # 데이터 타입 변환 (numpy -> Python 기본 타입)
        df_converted = df.astype({
            col: 'object'
            for col in df.select_dtypes(include=['int64', 'float64']).columns
        })

        conn = self.hook.get_conn()
        cursor = conn.cursor()

        try:
            # 컬럼 이름 및 플레이스홀더 생성
            columns = ', '.join([f'"{col}"' for col in df_converted.columns])
            placeholders = ', '.join(['%s'] * len(df_converted.columns))
            insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

            # DataFrame을 튜플 리스트로 변환
            records = [tuple(row) for row in df_converted.itertuples(index=False, name=None)]

            # 데이터 일괄 삽입
            cursor.executemany(insert_query, records)
            conn.commit()
            print(f"{table_name} 테이블에 {len(records)}개 레코드 삽입 완료")

        except Exception as e:
            conn.rollback()
            print(f"데이터 삽입 중 오류 발생: {e}")
            raise e

        finally:
            cursor.close()
            conn.close()

    def build_rack_query(
        self,
        start_time: str,
        end_time: str,
        battery_status_columns: List[str]
    ) -> str:
        """
        Rack 데이터 조회 쿼리 생성

        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            battery_status_columns: 배터리 상태 컬럼 리스트

        Returns:
            SQL 쿼리 문자열
        """
        battery_cols = ', '.join([f'bk."{col}"' for col in battery_status_columns])

        query = f"""
            SELECT DISTINCT
                rk."TIMESTAMP",
                rk."BANK_ID",
                rk."RACK_ID",
                rk."RACK_SOC",
                rk."RACK_CURRENT",
                rk."RACK_MAX_CELL_VOLTAGE",
                rk."RACK_MIN_CELL_VOLTAGE",
                rk."RACK_MAX_CELL_VOLTAGE_POSITION",
                rk."RACK_MIN_CELL_VOLTAGE_POSITION",
                rk."RACK_MAX_CELL_TEMPERATURE",
                rk."RACK_MIN_CELL_TEMPERATURE",
                rk."RACK_MAX_CELL_TEMPERATURE_POSITION",
                rk."RACK_MIN_CELL_TEMPERATURE_POSITION",
                {battery_cols}
            FROM (
                SELECT
                    "TIMESTAMP",
                    "BANK_ID",
                    "RACK_ID",
                    "RACK_SOC",
                    "RACK_CURRENT",
                    "RACK_MAX_CELL_VOLTAGE",
                    "RACK_MIN_CELL_VOLTAGE",
                    "RACK_MAX_CELL_VOLTAGE_POSITION",
                    "RACK_MIN_CELL_VOLTAGE_POSITION",
                    "RACK_MAX_CELL_TEMPERATURE",
                    "RACK_MIN_CELL_TEMPERATURE",
                    "RACK_MAX_CELL_TEMPERATURE_POSITION",
                    "RACK_MIN_CELL_TEMPERATURE_POSITION"
                FROM rack
                WHERE ("TIMESTAMP" BETWEEN '{start_time}' AND '{end_time}')
            ) AS rk
            INNER JOIN (
                SELECT
                    "TIMESTAMP",
                    {', '.join([f'"{col}"' for col in battery_status_columns])}
                FROM bank
                WHERE ("TIMESTAMP" BETWEEN '{start_time}' AND '{end_time}')
            ) AS bk ON rk."TIMESTAMP" = bk."TIMESTAMP"
            ORDER BY "TIMESTAMP" DESC;
        """
        return query
