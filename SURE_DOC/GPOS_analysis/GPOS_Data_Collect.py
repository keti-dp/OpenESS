import psutil
import paramiko
import os
import pandas as pd
import csv
from datetime import datetime
import schedule
import time

output_file = 'hw_idle_info'

# List to store accumulated data
# accumulated_data = []
file_name = os.getcwd() + '/output/' + datetime.strftime(datetime.now(), format='%Y%m%d_%H') + '_hw_idle_info.txt'
if not os.path.isdir(os.getcwd() + '/output/'):
    os.mkdir(os.getcwd() + '/output/')
savefile = open(file_name, 'w')
savefile.write('time, cpu, memory \n')
savefile.close()


def get_hw_idle_info():
    """CPU, 메모리 예비율 정보와 시간을 구한다"""
    rst = dict()  # 반환값 초기화

    cp = psutil.cpu_times_percent(interval=0.1, percpu=False)  # CPU 데이터

    # Add current timestamp to the dictionary
    # rst['time'] = datetime.strftime(datetime.now(), format='%Y%m%d_%H%M%S' )
    # rst['cpu'] = round(100 - cp.idle, 2)

    vm = psutil.virtual_memory()  # 메모리
    # rst['memory'] = vm.percent

    time = datetime.strftime(datetime.now(), format='%Y%m%d_%H%M%S.%f')
    cpu = str(round(100 - cp.idle, 2))
    memory = str(vm.percent)

    return [time, cpu, memory]


# def write_data_to_csv():
#     # Write accumulated data to CSV
#     with open(output_file+'_'+'.csv', 'a', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=accumulated_data[0].keys())
#         writer.writerows(accumulated_data)

#     accumulated_data.clear()

# def job():
#     data = get_hw_idle_info()
#     # accumulated_data.append(data)
#     print(f"Data has been accumulated at {datetime.now()}")
#     return data


def upload():
    global file_name
    connection = paramiko.transport.Transport(host)
    connection.connect(username=user, password=password)
    sftp = paramiko.SFTPClient.from_transport(connection)
    print('-----------------------------------------------------------------------------')
    print("Connect Server")

    # 업로드할 파일 설정
    sftp.put(file_name, root_remotepath + file_name[:-4] + '.csv')

    print('Done')
    sftp.close()

    connection.close()

    file_name = os.getcwd() + '/output/' + datetime.strftime(datetime.now(), format='%Y%m%d_%H%M') + '_hw_idle_info.txt'
    if not os.path.isdir(os.getcwd() + '/output/'):
        os.mkdir(os.getcwd() + '/output/')
    # accumulated_data = []


schedule.every(1).minutes.do(upload)

while True:
    file_name = os.getcwd() + '/output/' + datetime.strftime(datetime.now(), format='%Y%m%d_%H') + '_hw_idle_info.txt'
    if not os.path.isdir(os.getcwd() + '/output/'):
        os.mkdir(os.getcwd() + '/output/')

    data = get_hw_idle_info()
    # print(f"Data has been accumulated at {datetime.now()}")

    with open(file_name, 'a') as savefile:
        savefile.write(','.join(data) + '\n')

    # schedule.run_pending()
    # time.sleep(0.1)