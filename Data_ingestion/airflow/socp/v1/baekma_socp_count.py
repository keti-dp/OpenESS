from datetime import datetime, timedelta
from pytz import  timezone
from pprint import pprint
import pandas as pd
import os
import glob
import pendulum
from sqlalchemy import create_engine

from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy import DummyOperator

ORI_DATA_PATH = "/home/jpark/airflow/dags/calc_socp/dataset/baekma/"
WORK_DATA_PATH = "/home/jpark/airflow/dags/calc_socp/dataset/baekma/count/work_data/"
PREP_DATA_PATH = "/home/jpark/airflow/dags/calc_socp/dataset/baekma/count/prep_data/"
PERIOD = 1
SITE_NAME = "baekma"

def calc_past_days(**context):
    execution_date = context["execution_date"]

    seoul_tz = timezone('Asia/Seoul')
    execution_date_seoul = execution_date.astimezone(seoul_tz)

    print(f"Excution_data: {execution_date}")
    print(f"execution_date_seoul: {execution_date_seoul}")
      
    past_1days = execution_date_seoul.start_of('day').to_datetime_string()  
    end_time = execution_date_seoul.end_of('day').to_datetime_string()

    query_time = {"past_1days":past_1days, "end_time":end_time}
    print(query_time)
    context["task_instance"].xcom_push(key="query_time", value=query_time)


def prep_dataset(**context):
    load_file_name = context["task_instance"].xcom_pull(task_ids='calc_time', key='query_time')['past_1days']
    load_file_name = load_file_name[:10]
    df = pd.read_parquet(ORI_DATA_PATH + load_file_name + ".parquet")

    # UTC to ASIA/Seoul
    df.set_index('TIMESTAMP', inplace=True)
    new_index = df.index.tz_convert(timezone("Asia/Seoul"))
    df.index = new_index
    
    bank_list = df["BANK_ID"].unique()
    
    if not os.path.exists(WORK_DATA_PATH+load_file_name):
        os.makedirs(WORK_DATA_PATH+load_file_name)

    for bank_idx in bank_list:
        rack_list = df[df["BANK_ID"] == bank_idx]["RACK_ID"].unique()
        
        for rack_idx in rack_list:
            query_df = df.query(f"BANK_ID == {bank_idx} and RACK_ID == {rack_idx}")

            print(f"BANK_ID: {bank_idx}, RACK_ID: {rack_idx}")
            
            file_name = SITE_NAME + str(bank_idx) + "_" + str(rack_idx)
            query_df.to_csv(WORK_DATA_PATH + load_file_name + "/" +file_name + ".csv")


def calc_stateOfcellPosition(**context):
    load_file_name = context["task_instance"].xcom_pull(task_ids='calc_time', key='query_time')['past_1days']
    load_file_name = load_file_name[:10]

    merge_list = []
    file_list = glob.glob(WORK_DATA_PATH + load_file_name + "/" + "*")
    for file_name in file_list:
        df = pd.read_csv(file_name)
        
        df_standby = df[df["BATTERY_STATUS_FOR_CHARGE"] == 1]
        df_charge = df[df["BATTERY_STATUS_FOR_CHARGE"] == 2]
        df_discharge = df[df["BATTERY_STATUS_FOR_CHARGE"] == 3]

        status_list = [df_charge, df_discharge, df_standby]
        
        charge_status = 0
        for cs_dataframe in status_list:
            
            #result_dict["VALUE"] = {}

            for i in range(0, 100, 10):
                result_dict = {}
                result_dict["PERIOD"] = PERIOD
                result_dict["TIMESTAMP"] = context["task_instance"].xcom_pull(task_ids='calc_time', key='query_time')['past_1days']
                result_dict["BANK_ID"] = int(df["BANK_ID"].iloc[-1])
                result_dict["RACK_ID"] = int(df["RACK_ID"].iloc[-1])
                result_dict["CHARGE_STATUS"] = charge_status
                result_dict["SOC_RANGE"] = i
                
                for j in range(1, 241):
                    if charge_status == 0 or charge_status == 2:
                        condition = (cs_dataframe['RACK_SOC'] >= i) & (cs_dataframe['RACK_SOC'] < i+10) & (cs_dataframe['RACK_MAX_CELL_VOLTAGE_POSITION'] == j)
                    elif charge_status == 1:
                        condition = (cs_dataframe['RACK_SOC'] >= i) & (cs_dataframe['RACK_SOC'] < i+10) & (cs_dataframe['RACK_MIN_CELL_VOLTAGE_POSITION'] == j)

                    sub_df = cs_dataframe[condition]
                    count = len(sub_df)
                    result_dict["CELL_"+str(j)] = count
                
                merge_list.append(result_dict)
            charge_status += 1
    
    df_result = pd.DataFrame(merge_list)
    df_result = df_result.sort_values(by=['BANK_ID', 'RACK_ID', "CHARGE_STATUS", "SOC_RANGE"] ,ascending=True).reset_index(drop=True)
    df_result.to_csv(PREP_DATA_PATH + load_file_name +".csv")
    pprint(df_result)
                

def save_df_to_postgres(table_name, **context):
    load_file_name = context["task_instance"].xcom_pull(task_ids='calc_time', key='query_time')['past_1days']
    load_file_name = load_file_name[:10]

    hook  = PostgresHook(postgres_conn_id='ess_stats')
    engine = create_engine(hook.get_uri(), echo=False)
    
    df = pd.read_csv(PREP_DATA_PATH + load_file_name +".csv", index_col=0)
    df.to_sql(table_name, engine, if_exists='append', index=False)
        
    hook.get_conn().close()
    engine.dispose()
    
    


kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}

with DAG(dag_id='baekma_SoCP_count',
         default_args=default_args,
         start_date=datetime(2024, 1, 1, tzinfo=kst),
         schedule_interval=None,
         tags=['baekma', 'socp'],
         catchup=True
         ) as dag:

    start = DummyOperator(task_id="start")
    
    calc_past_time = PythonOperator(
        task_id='calc_time',
        python_callable=calc_past_days
    )

    preprocessing_dataset = PythonOperator(
        task_id='prep_dataset',
        python_callable=prep_dataset
    )

    calc_socp = PythonOperator(
        task_id='calc_socp',
        python_callable=calc_stateOfcellPosition
    )


    save_database = PythonOperator(
        task_id='save_database',
        python_callable=save_df_to_postgres,
        op_kwargs={
            "table_name": "ess_socp_count_oper4"
        }
    )

    end = DummyOperator(task_id="end")
    
    trigger_differencing = TriggerDagRunOperator(
        task_id = "trigger_differencing",
        trigger_dag_id = "baekma_SoCP_1st_differencing",
        execution_date = "{{ execution_date }}"
    )

    trigger_moving_average = TriggerDagRunOperator(
        task_id = "trigger_moving_average",
        trigger_dag_id = "baekma_SoCP_moving_average",
        execution_date = "{{ execution_date }}"
    )

start >> calc_past_time >> preprocessing_dataset >> calc_socp >> save_database >> [trigger_differencing, trigger_moving_average] >> end








