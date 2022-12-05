import pandas
import psycopg2
from matplotlib import pyplot as plt


class DataProcessing:
    @staticmethod
    def get_data_to_SQL(query_address: str,
                        query_id: str,
                        query_passwd: str,
                        data_num: int = 100) -> (bool, pandas.DataFrame, str):
        """
        서버로부터 ESS 측정 데이터 가져오기

        :param query_address:   SQL 쿼리 주소
        :param query_id:        SQL 접속 ID
        :param query_passwd:    SQL 접속 암호
        :param data_num:        불러올 데이터 인덱스 (기본값 : 100개)
        :return:                (처리 결과, DataFrame, 오류 메시지)
        """
        try:
            import pandas.io.sql as psql
        except ImportError:
            return False, None, 'Error : No Module'
        sql_connection = psycopg2.connect(f'postgres://{query_id}:{query_passwd}@{query_address}')
        query = f'select * from bank order by 1 desc limit {data_num}'
        data = psql.read_sql(query, sql_connection)
        sql_connection.close()
        return True, data, 'No Error'

    @staticmethod
    def get_data_to_csv(file_path) -> (bool, pandas.DataFrame, str):
        """
        측정 로그 데이터 불러오기

        :param file_path:   파일 경로
        :return:            (처리 결과, DataFrame, 오류 메시지)
        """
        try:
            import pandas
        except ImportError:
            return False, None, 'Error : No Module'
        return True, pandas.read_csv(file_path), 'No Error'

    @staticmethod
    def preprocessing(data: pandas.DataFrame,
                      run_normalization: bool = False,
                      run_standardization: bool = False) -> (pandas.DataFrame, list[str]):
        """
        데이터 표준화 및 정규화

        :param data:                입력 데이터
        :param run_normalization:   정규화 수행 여부
        :param run_standardization: 표준화 수행 여부
        :return:                    처리 완료된 데이터, 필터링 된 Column List
        """
        if run_normalization:
            pass
        columns = list(data.columns)
        columns = [column for column in columns if all([not column.endswith('Position'),
                                                        not column.startswith('Unnamed'),
                                                        not column == ' '])]
        for column in columns:
            try:
                if run_standardization:
                    data[column] = (data[column] - (data[column].mean())) / data[column].std()
                if run_normalization:
                    if not data[column].min() == data[column].max():
                        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
                    else:
                        data[column] = 1
            except TypeError:
                continue
        return data, columns
