import multiprocessing
import sys
import psycopg2
import pandas.io.sql as psql
import time
import math
import pandas as pd

def cal_sum(input_list):
    res = 0
    for i in input_list:
        res += i

    return res

def spliter(input_list):
    amount = math.floor(len(input_list)/num_pool)
    sub_list = []
    for i in range(num_pool):
        if i == num_pool-1:
            sub_list.append(input_list[amount*(num_pool-1):len(input_list)])
            break
        sub_list.append(input_list[amount*i:amount*(i+1)])

    return sub_list

if __name__ == '__main__':

    print('start')
    """Data Load"""
    # database connection
    # CONNECTION = "postgres://guest_user:####@1.1.1.1:1111/ESS_Operating_Site1"
    conn = psycopg2.connect(CONNECTION)
    cursor = conn.cursor()

    # ESS rack dataset load
    sql =  "SELECT * FROM rack " \
           "WHERE \"TIMESTAMP\" > \'2021-10-01 00:00:00\' and \"TIMESTAMP\" < \'2021-10-11 00:00:00\'"
    df_rack = psql.read_sql(sql, conn)
    dict_rack = df_rack.to_dict()

    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', 13)
    pd.set_option('display.width', None)

    # Check Dataset
    print('DataSize(len):', len(df_rack))
    print('DataSize(byte):', sys.getsizeof(df_rack))
    print(df_rack.head(20))
    print(df_rack.describe())

    # Setting target
    target_col = ['RACK_VOLTAGE']
    #target_col = ['RACK_VOLTAGE', 'RACK_CURRENT']
    num_pool = 8

    start = time.process_time()

    """Multi Threading"""
    for target in target_col:

        target_dataset = list(dict_rack[target_col[0]].values())

        pool = multiprocessing.Pool(processes=num_pool)
        sub_routine = spliter(target_dataset)
        results = pool.map(cal_sum, sub_routine)
        pool.close()
        pool.join()
        print('results:', results)

        avg = sum(results)/len(target_dataset)
        print('target:', target)
        print('avg:', avg)

    finish = time.process_time()
    print(f'Finished in {round(finish-start,2)} seconds(s)')