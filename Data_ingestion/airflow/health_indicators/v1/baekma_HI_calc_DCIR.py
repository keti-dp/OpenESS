from datetime import datetime, timedelta
from pytz import  timezone
from pprint import pprint
import pandas as pd
import pendulum

from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import BranchPythonOperator

DATA_SAVE_PATH = "/home/jpark/airflow/dags/calc_health_indicator/dataset/baekma/"
OPERATING_SITE = 4

def clac_past_days(**context):
    execution_date = context["execution_date"]

    seoul_tz = timezone('Asia/Seoul')
    execution_date_seoul = execution_date.astimezone(seoul_tz)

    print(f"Excution_data: {execution_date}")
    print(f"execution_date_seoul: {execution_date_seoul}")
      
    start_time = execution_date_seoul.start_of('day').to_datetime_string()
    end_time = execution_date_seoul.end_of('day').to_datetime_string()

    query_time = {"start_time":start_time, "end_time":end_time}
    print(query_time)
    context["task_instance"].xcom_push(key="query_time", value=query_time)
    


def calc_health_indicator_dcir(**context):
    # dag_conf = context["dag_run"].conf
    # filename = dag_conf.get("filename")
    query_time = context["task_instance"].xcom_pull(task_ids="clac_past_days", key="query_time")
    filename = str(query_time["start_time"])[:10]
    df = pd.read_parquet(DATA_SAVE_PATH + filename + ".parquet")

    result_value = {}
    bank_list = df["BANK_ID"].unique()
    for bank_id in bank_list:
        result_value[int(bank_id)] = {}  # bank_id에 대한 내부 딕셔너리 생성
        filtered_df = df.query(f"BANK_ID == {bank_id}")
        rack_list = filtered_df["RACK_ID"].unique()

        for rack_id in rack_list:          
            try:
                # rack_id에 대한 내부 딕셔너리 생성
                result_value[int(bank_id)][int(rack_id)] = {}  
                
                # 데이터가 부족할 때는 어떻게 할것인가...?
                standby_df = filtered_df.query(f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_RUN == 1 and BATTERY_STATUS_FOR_CHARGE == 1 ").reset_index(drop=True)
                cdcir_df = filtered_df.query(f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_RUN == 1 and BATTERY_STATUS_FOR_CHARGE == 2").reset_index(drop=True)
                ddcir_df = filtered_df.query(f"RACK_ID == {rack_id} and BATTERY_STATUS_FOR_RUN == 1 and BATTERY_STATUS_FOR_CHARGE == 3").reset_index(drop=True)
                
                # 'TIMESTAMP'를 index로 설정 및 중복제거
                standby_df = standby_df.set_index('TIMESTAMP')
                cdcir_df = cdcir_df.set_index('TIMESTAMP')
                ddcir_df = ddcir_df.set_index('TIMESTAMP')
                
                standby_df = standby_df.loc[~standby_df.index.duplicated(keep='first')]
                cdcir_df = cdcir_df.loc[~cdcir_df.index.duplicated(keep='first')]
                ddcir_df = ddcir_df.loc[~ddcir_df.index.duplicated(keep='first')]
                
                    
                # 휴지 전류 0 -> 충전 후 전류가 10 이상 인 시점의 DCIR
                min_time_cdcir = cdcir_df.index.min()
                charge_standby_df = standby_df[standby_df.index < min_time_cdcir]
                last_time_with_zero_current = charge_standby_df[charge_standby_df['RACK_CURRENT'] == 0].index[-1]
                
                filtered_cdcir_df = cdcir_df[(cdcir_df.index > last_time_with_zero_current) & (cdcir_df['RACK_CURRENT'] >= 10)]
                closest_time_with_current_above_10 = filtered_cdcir_df.index.min()
                
                cdcir_v_diff = cdcir_df.loc[closest_time_with_current_above_10, "RACK_VOLTAGE"] - standby_df.loc[last_time_with_zero_current, "RACK_VOLTAGE"]
                cdcir_i_diff = cdcir_df.loc[closest_time_with_current_above_10, "RACK_CURRENT"] - standby_df.loc[last_time_with_zero_current, "RACK_CURRENT"]
                
                cdcir = round(abs(cdcir_v_diff / cdcir_i_diff), 2)
                
                
                # 휴지 전류 0 -> 방전 후 전류가 -10 이하 인 시점의 DDCIR
                min_time_ddcir = ddcir_df.index.min()
                discharge_standby_df = standby_df[standby_df.index < min_time_ddcir]
                last_time_with_zero_current = discharge_standby_df[discharge_standby_df['RACK_CURRENT'] == 0].index[-1]
                
                filtered_ddcir_df = ddcir_df[(ddcir_df.index > last_time_with_zero_current) & (ddcir_df['RACK_CURRENT'] <= -10)]
                closest_time_with_current_above_10 = filtered_ddcir_df.index.min()

                
                ddcir_v_diff = ddcir_df.loc[closest_time_with_current_above_10, "RACK_VOLTAGE"] - standby_df.loc[last_time_with_zero_current, "RACK_VOLTAGE"]
                ddcir_i_diff = ddcir_df.loc[closest_time_with_current_above_10, "RACK_CURRENT"] - standby_df.loc[last_time_with_zero_current, "RACK_CURRENT"]

                ddcir = round(abs(ddcir_v_diff / ddcir_i_diff), 2)
                
                
                print(f"BANK_ID: {bank_id}, RACK_ID: {rack_id}, CDCIR: {cdcir}, DDCIR: {ddcir}")
                    
                result_value[int(bank_id)][int(rack_id)] = {
                    "OPERATING_SITE": OPERATING_SITE,
                    "TIMESTAMP": filename,
                    "CDCIR": cdcir,
                    "DDCIR": ddcir,
                }
            except ValueError:
                print(f"Error: {bank_id}_{rack_id} no data for DCIR calculation")
                continue  # 다음 rack_id로 계속 진행
            except IndexError:
                print(f"Error: {bank_id}_{rack_id} no data for DCIR calculation")
                continue  # 다음 rack_id로 계속 진행
            except KeyError:
                print(f"Error: {bank_id}_{rack_id} no data for DCIR calculation")
                continue  # 다음 rack_id로 계속 진행
            
    context["task_instance"].xcom_push(key="result", value=result_value)
    print(result_value)



def push_data_to_database(**context):
    result = context["task_instance"].xcom_pull(task_ids="calc_dcir", key="result")
    query_time = context["task_instance"].xcom_pull(task_ids="clac_past_days", key="query_time")

    pg_hook = PostgresHook(postgres_conn_id='ess_stats')
    
    conn = pg_hook.get_conn()
    cur = conn.cursor()

    for bank, rack in result.items():
        bank_id = bank

        for rack_id, value in rack.items():
            print(f"bank_id = {bank_id}, rack_id = {rack_id}, value = {value}")
            insert_query = """INSERT INTO health_indicator_dcir ("TIMESTAMP", 
                                                                "OPERATING_SITE", 
                                                                "BANK_ID", 
                                                                "RACK_ID", 
                                                                "CDCIR", 
                                                                "DDCIR") VALUES (%s, %s, %s, %s, %s, %s)"""
            try:
                cur.execute(insert_query, (query_time["start_time"], OPERATING_SITE, bank_id, rack_id, value["CDCIR"], value["DDCIR"]))
                conn.commit()
            except KeyError:
                print(f"Error: {bank_id}_{rack_id} no value(CDCIR, DDCIR)")
                continue
    
    cur.close()
    pg_hook.get_conn().close()


def check_data_for_push(**context):
    result_value = context["task_instance"].xcom_pull(task_ids='calc_dcir', key='result')
    print(result_value)

    # 각 bank_id 및 rack_id에 대한 데이터 확인
    for bank_id, racks in result_value.items():
        for rack_id, data in racks.items():
            # CDCIR 및 DDCIR 값이 모두 있는지 확인
            if 'CDCIR' in data and 'DDCIR' in data:
                return 'push_data'

    # 위 조건을 만족하는 데이터가 없으면 'end' 반환
    return 'end'


kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}

with DAG(dag_id='baekma_HI_calc_DCIR',
         default_args=default_args,
         start_date=datetime(2024, 5, 27, tzinfo=kst),
         schedule_interval=None,
         tags=['baekma', 'indicator', 'rack'],
         catchup=True
         ) as dag:

    start = DummyOperator(task_id="start")
    
    clac_past_time = PythonOperator(
        task_id='clac_past_days',
        python_callable=clac_past_days
    )

    calc_dcir = PythonOperator(
        task_id='calc_dcir',
        python_callable=calc_health_indicator_dcir
    )

    branch_operator = BranchPythonOperator(
        task_id='branch_check_data',
        python_callable=check_data_for_push,
        provide_context=True,
        dag=dag,
    )

    push_data = PythonOperator(
        task_id='push_data',
        python_callable=push_data_to_database
    )

    complete_data_push = DummyOperator(task_id="complete_data_push")
    end = DummyOperator(task_id="end")

start >> clac_past_time >> calc_dcir >> branch_operator
branch_operator >> push_data >> complete_data_push
branch_operator >> end