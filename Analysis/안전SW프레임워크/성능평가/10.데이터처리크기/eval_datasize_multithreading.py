from concurrent import futures
import sys
import psycopg2
import pandas.io.sql as psql
import time
import math

def cal_sum(input_list):
    res = 0
    for i in input_list:
        res += i

    return res

def spliter(input_list):
    amount = math.floor(len(input_list)/num_thread)
    sub_list = []
    for i in range(num_thread):
        if i == num_thread-1:
            sub_list.append(input_list[amount*(num_thread-1):len(input_list)])
            break
        sub_list.append(input_list[amount*i:amount*(i+1)])

    return sub_list

if __name__ == '__main__':

    """Data Load"""
    # database connection
    # CONNECTION = "postgres://guest_user:####@1.1.1.1:1111/ESS_Operating_Site1"
    conn = psycopg2.connect(CONNECTION)
    cursor = conn.cursor()

    # ESS rack dataset load
    sql = "SELECT * FROM rack limit 4000000"
    df_rack = psql.read_sql(sql, conn)
    dict_rack = df_rack.to_dict()

    # Check Dataset
    print(df_rack.head(5))
    print('DataSize(byte):', sys.getsizeof(df_rack))

    # Setting target
    target_col = ['RACK_VOLTAGE']
    #target_col = ['RACK_VOLTAGE', 'RACK_CURRENT']
    num_thread = 2

    start = time.perf_counter()

    """Multi Threading"""
    for target in target_col:

        target_dataset = list(dict_rack[target_col[0]].values())

        with futures.ThreadPoolExecutor() as executor:
            sub_routine = spliter(target_dataset)
            results = executor.map(cal_sum, sub_routine)

        avg = sum(results)/len(target_dataset)
        print('target:', target)
        print('avg:', avg)

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,2)} seconds(s)')
