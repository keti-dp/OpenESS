from datetime import datetime, timedelta
from pytz import  timezone
from pprint import pprint
import pandas as pd
import pendulum
import numpy as np
import json

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Dropout
from attention import Attention

from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy import DummyOperator

LOAD_DATA_PATH = "/home/jpark/airflow/dags/MaxVol_multistep_forecasting/dataset/panly/"
SAVE_DATA_PATH = "/home/jpark/airflow/dags/MaxVol_multistep_forecasting/dataset/panly/train/"
MODEL_PATH = "/home/jpark/airflow/dags/MaxVol_multistep_forecasting/model/"
SITE_NUM = 2 #panly
def calc_past_days(**context):
    execution_date = context["execution_date"]

    seoul_tz = timezone('Asia/Seoul')
    execution_date_seoul = execution_date.astimezone(seoul_tz)

    print(f"Excution_data: {execution_date}")
    print(f"execution_date_seoul: {execution_date_seoul}")
      
    start_time = (execution_date_seoul - pendulum.duration(days=13)).start_of('day').to_datetime_string()
    end_time = execution_date_seoul.end_of('day').to_datetime_string()

    query_time = {"start_time":start_time, "end_time":end_time}
    print(query_time)
    context["task_instance"].xcom_push(key="query_time", value=query_time)


def get_model():
    """
    어텐션 메커니즘을 가진 Keras LSTM 모델을 정의하고 반환합니다.

    Returns:
        tf.keras.Model: 컴파일된 Keras 모델.
    """
    
    #model = Sequential()
    input_layer = Input(shape=(2016, 5))
    lstm = LSTM(64, return_sequences=True, activation="tanh")(input_layer)
    dropout = Dropout(0.3)(lstm)
    att = Attention(64)(dropout)
    dence_4 = Dense(144)(att)
    output_layer = Dense(144)(dence_4)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model


def get_create_time(**context):
    start_time = datetime.strptime(str(datetime.now())[:11] + "00:00:00", "%Y-%m-%d %H:%M:%S")

    time_ = []

    for i in range(0, 24 * 6):
        current_time = start_time + i * timedelta(minutes=10)
        time_.append(str(current_time))
    
    context["task_instance"].xcom_push(key="create_time", value=time_)


def prep_dataset(**context):
    query_time = context["task_instance"].xcom_pull(task_ids="calc_past_time", key="query_time")
    start_date = query_time["start_time"][:10]
    end_date = query_time["end_time"][:10]

    data_range = pd.date_range(start_date, end_date, freq='D')
    date_list = [date.strftime('%Y-%m-%d') for date in data_range]

    sum_df = pd.DataFrame()
    for date in date_list:
        df = pd.read_feather(LOAD_DATA_PATH + date + ".feather")
        sum_df = pd.concat([sum_df, df])
    
    sum_df.reset_index(drop=True, inplace=True)
    
    # 학습 데이터 셋을 위한 작업 : 10분 단위로 데이터 획득
    list_ = []
    for i in range(0, 6):
        try:
            f_df = sum_df.query(f"second == 0 and minute == {i*10}")
            list_.append(f_df)

        except Exception as e:
            print("Error: ", e)
            continue
    
    result_df = pd.concat(list_, axis=0).sort_values("TIMESTAMP").drop_duplicates(subset=["TIMESTAMP", "BANK_ID", "RACK_ID"]).reset_index(drop=True)
    result_df["BANK_ID"] = result_df["BANK_ID"].astype(int)
    result_df["RACK_ID"] = result_df["RACK_ID"].astype(int)

    # 데이터셋 저장
    now = datetime.now().strftime('%Y-%m-%d')
    result_df.to_feather(SAVE_DATA_PATH + now + ".feather")
    print(result_df)


def predict_with_lstm_attention(**context):
    now = datetime.now().strftime('%Y-%m-%d')
    df = pd.read_feather(SAVE_DATA_PATH + now + ".feather")
    
    lstm_att_model = get_model()
    lstm_att_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_att_model.load_weights(f'{MODEL_PATH}BestAttModel_{SITE_NUM}.h5')

    result_dict = {}
    for bank_id in list(df["BANK_ID"].unique()):
        result_dict[int(bank_id)] = {}
        bank_df = df.query(f" BANK_ID == {bank_id}")

        for rack_id in list(bank_df["RACK_ID"].unique()):
            result_dict[bank_id][int(rack_id)] = {}
            rack_df = bank_df.query(f" RACK_ID == {rack_id}")

            rack_df.index = rack_df["TIMESTAMP"]

            rack_df = rack_df.drop(columns=["BANK_ID", "RACK_ID", "TIMESTAMP", "second", "minute"], axis=1)

            # min-max normalization
            df_norm = (rack_df-rack_df.mean()) / (rack_df.max() - rack_df.min())
            df_set = df_norm.to_numpy()        
            df_set = np.expand_dims(df_set, axis=0)

            # reverse normalized
            min_ = df["RACK_MAX_CELL_VOLTAGE"].min()
            max_ = df["RACK_MAX_CELL_VOLTAGE"].max()
            mean_ = df["RACK_MAX_CELL_VOLTAGE"].mean()

            prediction = lstm_att_model.predict(df_set)
            att_result = (prediction[0]*(max_ - min_) + mean_).tolist()
            att_result = [round(float(i), 3) for i in att_result]  # numpy.float32를 Python float으로 변환

            result_dict[bank_id][rack_id] = att_result

    print(result_dict)
    
    context["task_instance"].xcom_push(key="predict", value=result_dict)


def push_data_to_database(**context):
    time = context["task_instance"].xcom_pull(task_ids="generate_time_10minute_intervals", key="create_time")
    pred = context["task_instance"].xcom_pull(task_ids="data_inference", key="predict")

    save_time = datetime.strptime(str(datetime.now())[:10] + " 00:00:00", "%Y-%m-%d %H:%M:%S")

    pg_hook = PostgresHook(postgres_conn_id='ess_stats')
    
    conn = pg_hook.get_conn()
    cur = conn.cursor()

    for bank, rack in pred.items():
        bank_id = bank

        for rack_id, value in rack.items():
            values = {"lstm_attention": value, "time": time}
            print(f"bank_id = {bank_id}, rack_id = {rack_id}, value = {value}")
            insert_query = """INSERT INTO multi_step_forecasting_maxvol ("time", 
                                                                    "operating_site_id", 
                                                                    "bank_id", 
                                                                    "rack_id", 
                                                                    "values") VALUES (%s, %s, %s, %s, %s)"""
            cur.execute(insert_query, (save_time, SITE_NUM, bank_id, rack_id, json.dumps(values)))
            conn.commit()
    
    cur.close()
    pg_hook.get_conn().close()


kst = pendulum.timezone("Asia/Seoul")
default_args = {
    'owner' : 'jwpark',
}


with DAG(dag_id='panly_1day_voltage_forecasting',
         default_args=default_args,
         start_date=datetime(2023, 9, 15, tzinfo=kst),
         schedule_interval='10 01 * * *',
         tags=['forecasting'],
         catchup=False
         ) as dag:
    start = DummyOperator(task_id = "start")

    calc_past_time = PythonOperator(
        task_id='calc_past_time',
        python_callable=calc_past_days
    )

    data_preprocessing = PythonOperator(
        task_id='data_preprocessing',
        python_callable=prep_dataset
    )

    generate_time_10minute_intervals = PythonOperator(
        task_id='generate_time_10minute_intervals',
        python_callable=get_create_time
    )

    data_inference = PythonOperator(
        task_id='data_inference',
        python_callable=predict_with_lstm_attention
    )

    push_data = PythonOperator(
        task_id='push_data',
        python_callable=push_data_to_database
    )

    end = DummyOperator(task_id = "end")


# Task 간의 실행 순서를 정의합니다.
start >> calc_past_time >> data_preprocessing >> [generate_time_10minute_intervals, data_inference] >> push_data >> end