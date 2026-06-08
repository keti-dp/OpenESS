from datetime import datetime, timedelta
from pytz import  timezone
import pandas as pd
import pendulum
import numpy as np
import os
from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy import DummyOperator

DATA_SAVE_PATH = "/home/jpark/airflow/dags/calc_snu_indicator/dataset/panly/"
OPERATING_SITE = 2


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
    


def calc_snu_voltage_relax(**context):
    dag_conf = context["dag_run"].conf
    filename = dag_conf.get("filename")
    
    query_time = context["task_instance"].xcom_pull(task_ids='clac_past_days', key="query_time")
    start_time = datetime.strptime(query_time["start_time"].split(" ")[0], "%Y-%m-%d")
    
    # start_time에서 90일을 빼기
    adjusted_start_time = start_time - timedelta(days=90)
    date_range = pd.date_range(start=adjusted_start_time, end=start_time)

    results = []
    for i in range(1, len(date_range)):
        
        try:
            before_df = pd.read_parquet(DATA_SAVE_PATH + date_range[i-1].strftime('%Y-%m-%d') + ".parquet")
            after_df = pd.read_parquet(DATA_SAVE_PATH + date_range[i].strftime('%Y-%m-%d') + ".parquet")

            if (len(before_df) < 60000) or (len(after_df) < 60000):
                print("Data is insufficient!")
                print(f"{date_range[i-1].strftime('%Y-%m-%d')}: {len(before_df)}")
                print(f"{date_range[i].strftime('%Y-%m-%d')}: {len(after_df)}")
                continue
            
            for bank_id in after_df["BANK_ID"].unique():
                before_bank_df = before_df[before_df["BANK_ID"] == bank_id]
                after_df_bank_df = after_df[after_df["BANK_ID"] == bank_id]

                for rack_id in after_df_bank_df["RACK_ID"].unique():
                    # 방전후 휴지기 데이터 획득
                    before_filtered_df = before_bank_df[before_bank_df["RACK_ID"] == rack_id].reset_index(drop=True)
                    first_discharge_index = before_filtered_df[before_filtered_df['BATTERY_STATUS_FOR_DISCHARGE'] == 1].index[0]
                    dischage_to_stanby_df = before_filtered_df.loc[first_discharge_index:].query('BATTERY_STATUS_FOR_STANDBY == 1').reset_index(drop=True)
        
                    # 충전 전 휴지기의 데이터 추출
                    after_filtered_df = after_df_bank_df[after_df_bank_df["RACK_ID"] == rack_id].reset_index(drop=True)
                    first_charge_index = after_filtered_df[after_filtered_df['BATTERY_STATUS_FOR_CHARGE'] == 1].index[0]
                    stanby_to_charge_df = after_filtered_df.loc[:first_charge_index].query('BATTERY_STATUS_FOR_STANDBY == 1').reset_index(drop=True)

                    combined_df = pd.concat([dischage_to_stanby_df, stanby_to_charge_df], ignore_index=True)

                    rack_max_cell_voltage = combined_df["RACK_MAX_CELL_VOLTAGE"].to_numpy()

                    temp = np.array(rack_max_cell_voltage) - np.min(rack_max_cell_voltage)
                    temp = temp / np.max(temp)
                    area = np.sum(temp) / len(temp)
                    
                    results.append({
                        "TIMESTAMP": date_range[i].strftime('%Y-%m-%d'),
                        "BANK_ID": int(bank_id),
                        "RACK_ID": int(rack_id),
                        "VALUE": float(round(area, 4))
                    })
        
        except Exception as e:
            print(f"Failed to read data : {e}")
            continue
    
    result_list = []
    
    results_df = pd.DataFrame(results)
    
    for bank_id in results_df["BANK_ID"].unique():
        bank_df = results_df[results_df["BANK_ID"] == bank_id]

        for rack_id in bank_df["RACK_ID"].unique():
            rack_df = bank_df[bank_df["RACK_ID"] == rack_id]
            result_list.append((int(bank_id), 
                                int(rack_id), 
                                float(round(rack_df["VALUE"].mean(), 4)), 
                                float(round(rack_df["VALUE"].var(), 4))
                            ))
        
        print(result_list)

    context["task_instance"].xcom_push(key="result", value=result_list)


def push_data_to_database(**context):
    result = context["task_instance"].xcom_pull(task_ids="calc_voltage_relax", key="result")
    query_time = context["task_instance"].xcom_pull(task_ids="clac_past_days", key="query_time")


    pg_hook = PostgresHook(postgres_conn_id='ess_stats')
    
    conn = pg_hook.get_conn()
    cur = conn.cursor()

    for item in result:
        print(f"bank_id = {item[0]}, rack_id = {item[1]}, voltage_mean = {item[2]}, voltage_var = {item[3]}")
        insert_query = """INSERT INTO snu_voltage_relax ("TIMESTAMP", 
                                                        "OPERATING_SITE", 
                                                        "BANK_ID", 
                                                        "RACK_ID", 
                                                        "VOLTAGE_MEAN",
                                                        "VOLTAGE_VAR") VALUES (%s, %s, %s, %s, %s, %s)"""
        cur.execute(insert_query, (query_time["start_time"], OPERATING_SITE, item[0], item[1], item[2], item[3]))
        conn.commit()
    
    cur.close()
    pg_hook.get_conn().close()

def manage_files(**context):
    files = os.listdir(DATA_SAVE_PATH)
    parquet_files = [f for f in files if f.endswith('.parquet')]
    
    if len(parquet_files) > 90:
        # 파일들을 오래된 순으로 정렬
        parquet_files.sort(key=lambda x: os.path.getmtime(os.path.join(DATA_SAVE_PATH, x)))
        # 가장 오래된 파일 삭제
        os.remove(os.path.join(DATA_SAVE_PATH, parquet_files[0]))
        print(f"Removed oldest file: {parquet_files[0]}")


kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}

with DAG(dag_id='panly_snu_voltage_relax',
         default_args=default_args,
         start_date=datetime(2024, 1, 1, tzinfo=kst),
         schedule_interval=None,
         tags=['panly', 'indicator', 'snu'],
         catchup=True
         ) as dag:

    start = DummyOperator(task_id="start")
    
    clac_past_time = PythonOperator(
        task_id='clac_past_days',
        python_callable=clac_past_days
    )


    calc_voltage_relax = PythonOperator(
        task_id='calc_voltage_relax',
        python_callable=calc_snu_voltage_relax
    )

    push_data = PythonOperator(
        task_id='push_data',
        python_callable=push_data_to_database
    )

#     manage_files_operator = PythonOperator(
#     task_id='manage_files',
#     python_callable=manage_files,
#     dag=dag
# )
    
    end = DummyOperator(task_id="end")


#start >> clac_past_time >> calc_voltage_relax >> push_data >> manage_files_operator >> end
start >> clac_past_time >> calc_voltage_relax >> push_data >> end




