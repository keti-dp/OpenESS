from datetime import datetime, timedelta
from pytz import  timezone
from pprint import pprint
import pandas as pd
import os
import glob
import json
import copy
import pendulum

from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy import DummyOperator

LOAD_DATA_PATH = "/home/jpark/airflow/dags/calc_socp/dataset/panly/"
PERIOD_DATA_PATH = "/home/jpark/airflow/dags/calc_socp/dataset/panly/work_data/info/period_dataset/"
PREP_DATA_PATH = "/home/jpark/airflow/dags/calc_socp/dataset/panly/work_data/info/prep_dataset/"

def calc_past_days(**context):
    execution_date = context["execution_date"]

    seoul_tz = timezone('Asia/Seoul')
    execution_date_seoul = execution_date.astimezone(seoul_tz)

    print(f"Excution_data: {execution_date}")
    print(f"execution_date_seoul: {execution_date_seoul}")
      
    past_1days = execution_date_seoul.start_of('day').to_datetime_string()
    past_7days = (execution_date_seoul - pendulum.duration(days=6)).start_of('day').to_datetime_string()
    past_30days = (execution_date_seoul - pendulum.duration(days=29)).start_of('day').to_datetime_string()
    
    end_time = execution_date_seoul.end_of('day').to_datetime_string()

    query_time = {"past_30days":past_30days, "past_7days":past_7days, "past_1days":past_1days, "end_time":end_time}
    print(query_time)
    context["task_instance"].xcom_push(key="query_time", value=query_time)
    

def get_dataset(inf, **context):
    start = context["task_instance"].xcom_pull(task_ids='calc_time', key='query_time')[inf]
    end  = context["task_instance"].xcom_pull(task_ids='calc_time', key='query_time')['end_time']
    f_name = end[:10]
    date_range = pd.date_range(start=start, end=end).strftime('%Y-%m-%d').tolist()
    
    list_of_df = []
    for date in date_range:
        try:
            df = pd.read_feather(LOAD_DATA_PATH + date + ".feather")
            list_of_df.append(df)
        except FileNotFoundError:
            print(f"{date} No data!")
            continue
        
    df_accum = pd.concat(list_of_df).reset_index(drop=True)
    print(df_accum)

    if not os.path.exists(PERIOD_DATA_PATH+f_name):
        os.makedirs(PERIOD_DATA_PATH+f_name)

    bank_list = df["BANK_ID"].unique()
    for bank in bank_list:
        query_bank_df = df_accum.query(f"BANK_ID == {bank}")
        rack_list = query_bank_df["RACK_ID"].unique()

        for rack in rack_list:
            query_rack_df = query_bank_df.query(f"RACK_ID == {rack}").reset_index(drop=True)
            file_name = inf + "_" + str(bank) + "_" + str(rack)
            query_rack_df.to_feather(PERIOD_DATA_PATH + f_name + "/" + file_name + ".feather")


def calc_stateOfcellPosition(file, period, site, **context):
    start = context["task_instance"].xcom_pull(task_ids='calc_time', key='query_time')["past_1days"]
    end = context["task_instance"].xcom_pull(task_ids='calc_time', key='query_time')["end_time"]
    f_name = end[:10]

    result_dict = {}
    tmp_list = []
    result_dict["OPERATING_SITE"] = site
    result_dict["PERIOD"] = period

    file_list = glob.glob(PERIOD_DATA_PATH + f_name + "/" + file + "*")
    for file_name in file_list:
        df = pd.read_feather(file_name)
        
        result_dict["BANK_ID"] = str(df["BANK_ID"].iloc[-1])
        result_dict["RACK_ID"] = str(df["RACK_ID"].iloc[-1])
        
        df_charge = df[df["BATTERY_STATUS_FOR_CHARGE"] == 1]
        df_discharge = df[df["BATTERY_STATUS_FOR_DISCHARGE"] == 1]
        df_standby = df[df["BATTERY_STATUS_FOR_STANDBY"] == 1]

        status_list = [df_charge, df_discharge, df_standby]
        
        k = 0
        for cs_dataframe in status_list:
            result_dict["CHARGE_STATUS"] = k
            #result_dict["VALUE"] = {}

            for i in range(0, 100, 10):
                result_dict["SOC_RANGE"] = i
                position_dict = {}

                for j in range(1, 241):
                    condition = (cs_dataframe['RACK_SOC'] >= i) & (cs_dataframe['RACK_SOC'] < i+10) & (cs_dataframe['RACK_MAX_CELL_VOLTAGE_POSITION'] == j)
                    sub_df = cs_dataframe[condition]
                    count = len(sub_df)
                    voltage_mean = round(sub_df['RACK_MAX_CELL_VOLTAGE'].mean(), 3)
                    temp_mean = round(sub_df['RACK_MAX_CELL_TEMPERATURE'].mean(), 3)
                    current_mean = round(sub_df['RACK_CURRENT'].mean(), 3)
                    position_dict[str(j)] = {"count": count, "voltage": voltage_mean, "temp": temp_mean, "current": current_mean}

                # 'count'가 0인 요소 제거
                filtered_data = {key: value for key, value in position_dict.items() if value['count'] != 0}                                
                result_dict["VALUE"] = json.dumps(filtered_data)
                tmp_list.append(copy.deepcopy(result_dict))
            k += 1
    result_df = pd.DataFrame(tmp_list)
    result_df = result_df.sort_values(by=['BANK_ID', 'RACK_ID', 'CHARGE_STATUS', 'SOC_RANGE'] ,ascending=True).reset_index(drop=True)

    result_df["TIMESTAMP"] = start

    if not os.path.exists(PREP_DATA_PATH+f_name):
        os.makedirs(PREP_DATA_PATH+f_name)

    result_df.to_csv(PREP_DATA_PATH + f_name + "/" +file + ".csv")


def push_data_to_database(table_name, **context):
    end = context["task_instance"].xcom_pull(task_ids='calc_time', key='query_time')["end_time"]
    f_name = end[:10]

    file_list = sorted(os.listdir(PREP_DATA_PATH + f_name + "/"))
        
    pg_hook = PostgresHook(postgres_conn_id='ess_stats')
    
    conn = pg_hook.get_conn()
    cur = conn.cursor()
    
    for file_name in file_list:
        df = pd.read_csv(PREP_DATA_PATH + f_name+ "/" + file_name, index_col = 0)
        df_dict = df.to_dict()

        for i in range(len(df)):
            insert_query = """INSERT INTO ess_socp ("TIMESTAMP", "OPERATING_SITE", "BANK_ID", "RACK_ID", "PERIOD", "CHARGE_STATUS", "SOC_RANGE", "VALUE") VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
            cur.execute(insert_query, (df_dict["TIMESTAMP"][i], df_dict["OPERATING_SITE"][i], df_dict["BANK_ID"][i], df_dict["RACK_ID"][i], df_dict["PERIOD"][i], df_dict["CHARGE_STATUS"][i], df_dict["SOC_RANGE"][i], df_dict["VALUE"][i]))
            conn.commit()
    
    cur.close()
    pg_hook.get_conn().close()


kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}

with DAG(dag_id='panly_info_socp',
         default_args=default_args,
         start_date=datetime(2023, 11, 21, tzinfo=kst),
         schedule_interval=None,
         tags=['panly', 'socp'],
         catchup=True
         ) as dag:

    start = DummyOperator(task_id="start")
    
    calc_past_time = PythonOperator(
        task_id='calc_time',
        python_callable=calc_past_days
    )

    getDataset_past_1days = PythonOperator(
        task_id='getDataset_past_1days',
        python_callable=get_dataset,
        provide_context=True,
        op_kwargs={
            "inf": "past_1days"
        }
    )

    getDataset_past_7days = PythonOperator(
        task_id='getDataset_past_7days',
        python_callable=get_dataset,
        provide_context=True,
        op_kwargs={
            "inf": "past_7days"
        }
    )

    getDataset_past_30days = PythonOperator(
        task_id='getDataset_past_30days',
        python_callable=get_dataset,
        provide_context=True,
        op_kwargs={
            "inf": "past_30days"

        }
    )

    calc_SoCP_1days = PythonOperator(
        task_id='calc_SoCP_1days',
        python_callable=calc_stateOfcellPosition,
        op_kwargs={
            "file": "past_1days",
            "period": 1,
            "site" : 2
        }
    )

    calc_SoCP_7days = PythonOperator(
        task_id='calc_SoCP_7days',
        python_callable=calc_stateOfcellPosition,
        op_kwargs={
            "file": "past_7days",
            "period": 7,
            "site" : 2
        }
    )

    calc_SoCP_30days = PythonOperator(
        task_id='calc_SoCP_30days',
        python_callable=calc_stateOfcellPosition,
        op_kwargs={
            "file": "past_30days",
            "period": 30,
            "site" : 2
        }
    )

    push_data = PythonOperator(
        task_id='push_data',
        python_callable=push_data_to_database,
        op_kwargs={
            "table_name": "ess_socp"
        }
    )

    end = DummyOperator(task_id="end")

start >> calc_past_time
calc_past_time >> getDataset_past_1days >> calc_SoCP_1days >> push_data >> end
calc_past_time >> getDataset_past_7days >> calc_SoCP_7days >> push_data >> end
calc_past_time >> getDataset_past_30days >> calc_SoCP_30days >> push_data >> end
