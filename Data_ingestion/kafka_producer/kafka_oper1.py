#!/bin/python

"""
kafka_oper1.py : 태양광 ESS 데이터 수집을 위한 코드 (카프카 producer)

        ---------------------------------------------------------------------------
        Copyright(C) 2022, 윤태일 / KETI / taeil777@keti.re.kr

        아파치 라이선스 버전 2.0(라이선스)에 따라 라이선스가 부여됩니다.
        라이선스를 준수하지 않는 한 이 파일을 사용할 수 없습니다.
        다음에서 라이선스 사본을 얻을 수 있습니다.

        http://www.apache.org/licenses/LICENSE-2.0

        관련 법률에서 요구하거나 서면으로 동의하지 않는 한 소프트웨어
        라이선스에 따라 배포되는 것은 '있는 그대로' 배포되며,
        명시적이든 묵시적이든 어떠한 종류의 보증이나 조건도 제공하지 않습니다.
        라이선스에 따른 권한 및 제한 사항을 관리하는 특정 언어는 라이선스를 참조하십시오.
        ---------------------------------------------------------------------------

최신 테스트 버전 : 1.0.0 ver
최신 안정화 버전 : 1.0.0 ver

시온유 태양광 ESS데이터 수집을 위한 코드입니다.

구축된 kafka cluster에서 producer를 생성하여 modbus TCP통신을 통해 ESS 운영사이트에서 데이터를 저장합니다.

전체적인 코드에 대한 설명은 https://github.com/keti-dp/OpenESS 에서 확인하실 수 있습니다.
       

"""


from pyModbusTCP.client import ModbusClient
import time
from kafka import KafkaProducer
from json import dumps
from datetime import datetime
from multiprocessing import Process
import threading
from pytz import timezone
import numpy as np
import logs
import os
import threading


class ESS_Modbus:

    # 기본 클라이언트 설정
    def __init__(self):
        try:
            connection_logger = logs.get_logger(
                "operation1", log_path, "operation1.json"
            )
            self.client = ModbusClient("모드버스주소", 40001, unit_id=1)
            self.client.open()
        except Exception as e:
            log_massage = """Connection Error : {error}""".format(error=e)
            connection_logger.error(log_massage)

    # 클라이언트 세팅
    def client_set(self, IP, PORT, ID):
        self.client = ModbusClient(IP, PORT, unit_id=ID)

    # 단순 데이터 파싱
    def data_parsing(self, partID):

        try:
            # 파싱로그
            parsing_logger = logs.get_logger("operation1", log_path, "operation1.json")

            # 인풋레지스터의 경우 최대 125개밖에 못가져오기때문에 BMS데이터의 경우 수정이 필요함
            if partID == 1:  # BMS1
                start_address = 0
                end_address = 137  # 개
            elif partID == 2:  # BMS2
                start_address = 200
                end_address = 266  # 개
            elif partID == 3:  # BMS2
                start_address = 500
                end_address = 89
            elif partID == 4:  # BMS2
                start_address = 600
                end_address = 5

            # 파싱데이터 리스트
            data = []

            # 데이터 파싱

            # ETC의 경우 604번의 값은 쓰레기 데이터
            if partID == 4:
                data = self.client.read_input_registers(start_address, end_address)
                data.pop(3)

            # BMS1, 2의 경우 데이터 파싱 개수를 초과하기때문에 나눠서 처리
            elif partID == 1:
                data1 = self.client.read_input_registers(start_address, 125)
                data2 = self.client.read_input_registers(125, 12)
                data = data1 + data2

            elif partID == 2:
                data1 = self.client.read_input_registers(start_address, 125)
                data2 = self.client.read_input_registers(325, 125)
                data3 = self.client.read_input_registers(450, 16)
                data = data1 + data2 + data3
            elif partID == 3:
                data = self.client.read_input_registers(start_address, end_address)

            # parsing_logger.info("part %s data parsing success", partID)

        except Exception as e:

            log_massage = """part {partID} data parsing error : {error}""".format(
                partID=partID, error=e
            )
            parsing_logger.error(log_massage)

        # 데이터 전처리
        result = self.data_preprocessing(partID, data)

        # 뱅크 아이디 추가
        if partID == 3 or partID == 4:
            BANK_ID = 1
            data.insert(0, BANK_ID)
        # dtype("uint16")
        # for i in range(len(result)):
        #     print("registernum :", start_address + i, "value :", result[i])
        # print("type :", type(result[i]))

        # for i in range(len(result)):
        #     print(i+200, result[i])

        self.client.close()

        return list(result)

    # 데이터 전처리 (스케일팩터, 범위 확인)
    def data_preprocessing(self, partID, data):

        preprocessing_logger = logs.get_logger(
            "operation1", log_path, "operation1.json"
        )

        try:
            if partID == 1:  # BMS1
                scale_factor_file_path = file_path + "scalefactor_BMS1.txt"
                value_range_file_path = file_path + "range_BMS1.txt"
            elif partID == 2:  # BMS2  -> boolean
                scale_factor_file_path = file_path + "scalefactor_BMS2.txt"
                # BMS2는 0 1값밖에 없기 때문에 인트형으로 변경
                for i in range(len(data)):
                    data[i] = int(data[i])

            elif partID == 3:  # PCS
                scale_factor_file_path = file_path + "scalefactor_PCS.txt"
            elif partID == 4:  # ETC
                scale_factor_file_path = file_path + "scalefactor_ETC.txt"
                value_range_file_path = file_path + "range_ETC.txt"

            # 스케일 팩터파일
            scale_factor_file = open(scale_factor_file_path, "r")
            scale_factor = scale_factor_file.readlines()

            # 음수 변환
            for i in range(len(data)):
                if data[i] <= 32767:
                    pass
                elif 32767 < data[i] < 65536:
                    data[i] = -(65535 - data[i] + 1)

            # 파트 2, 3
            if partID == 2:
                for i in range(len(data)):
                    if self.in_range(data[i], 0, 3):
                        pass
                    else:
                        print("partID : ", partID)
                        print("번호 : ", i)
                        print("범위를 벗어난 값 입니다.", data[i])
                        # preprocessing_logger.warning(
                        #     "An out-of-range value occurs at address %s of part %s : %s",
                        #     i,
                        #     partID,
                        #     data[i],
                        # )
                scale_factor_file.close()
                return data

            elif partID == 3:
                pass
            else:
                value_range_file = open(value_range_file_path, "r")
                value_range = value_range_file.readlines()

            # 스케일 팩터 적용
            for i in range(len(data)):
                scale_factor[i] = scale_factor[i].strip("\n")  # 스케일팩터 줄바꿈 문자 제거
                data[i] = int(data[i]) * float(scale_factor[i])  # 스케일팩터 적용
                data[i] = float("{:.3f}".format(data[i]))  # 소수점 한자리 적용

            # 값 범위(레인지) 판별
            # PCS의 경우 값 범위가 없기때문에 그냥 리턴
            if partID == 3:
                scale_factor_file.close()
                # preprocessing_logger.info("part %s data preprocessing success", partID)
                return data

            for i in range(len(data)):
                value_range[i] = value_range[i].strip("\n")  # 레인지 줄바꿈 문자 제거
                min_value = float(value_range[i].split()[0])
                max_value = float(value_range[i].split()[2])

                if self.in_range(data[i], min_value, max_value):
                    pass
                else:
                    print("partID : ", partID)
                    print("번호 : ", i)
                    print("범위를 벗어난 값 입니다.", data[i])
                    # preprocessing_logger.warning(
                    #     "An out-of-range value occurs at address %s of part %s : %s",
                    #     i,
                    #     partID,
                    #     data[i],
                    # )

            scale_factor_file.close()
            value_range_file.close()

            # preprocessing_logger.info("part %s data preprocessing success", partID)

            return data
        except Exception as e:
            log_massage = """part {partID} data preprocessing error : {error}""".format(
                partID=partID, error=e
            )
            preprocessing_logger.error(log_massage)

    # 최대값 최소값 사이에 존재하는지 판별
    def in_range(self, value, min, max):
        return min <= value <= max if max >= min else max <= value <= min


# BMS1,2 데이터를 조작하기 위한 메서드
def data_manipulation(BMS1, BMS2):

    # BANK 데이터 만들기

    BANK_ID = 1

    BMS1_data1 = BMS1[:9]
    BMS1_data2 = BMS1[9:]
    BMS2_data1 = BMS2[:34]
    BMS2_data2 = BMS2[34:]

    BMS1_data1.insert(0, BANK_ID)

    # 0:정상 1:이상으로 치환 221~229, 239, 268, 297 등 수정요망

    for i in range(9):
        if BMS2_data1[21 + i] == 0:
            BMS2_data1[21 + i] = 1
        elif BMS2_data1[21 + i] == 1:
            BMS2_data1[21 + i] = 0
    if BMS2_data1[-1] == 0:
        BMS2_data1[-1] = 1
    elif BMS2_data1[-1] == 1:
        BMS2_data1[-1] = 0

    # Rack status for run 에 대한게 수정 필요
    # Rack 개수가 8이니까 8반복
    for i in range(8):
        if BMS2_data2[5 + 29 * i] == 0:
            BMS2_data2[5 + 29 * i] = 1
        elif BMS2_data2[5 + 29 * i] == 1:
            BMS2_data2[5 + 29 * i] = 0

    BANK = BMS1_data1 + BMS2_data1

    # print("BANK : ", BANK)

    # RACK 데이터 1~8까지 만들기
    Rack_list = []
    for i in range(1, 9):
        temp1 = BMS1_data2[0 + 16 * (i - 1) : 16 * i]
        temp2 = BMS2_data2[0 + 29 * (i - 1) : 29 * i]

        temp3 = temp1 + temp2
        temp3.insert(0, i)  # BANK, RACK ID 추가
        temp3.insert(0, BANK_ID)  # BANK, RACK ID 추가
        Rack_list.append(temp3)
    # print("RACK : ", Rack_list)

    return BANK, Rack_list


def main():

    seoultime = datetime.now(timezone("asia/seoul")).replace(microsecond=0)

    # 운영사이트
    operation_site = "operation1"

    print(seoultime)

    start = time.time()

    test1 = ESS_Modbus()
    test2 = ESS_Modbus()
    test3 = ESS_Modbus()
    test4 = ESS_Modbus()

    list1 = test1.data_parsing(1)
    list2 = test2.data_parsing(2)
    list3 = test3.data_parsing(3)
    list4 = test4.data_parsing(4)

    #    print("list1 : ", list1)
    #    print("list2 : ", list2)
    #    print("list3 : ", list3)
    #    print("list4 : ", list4)

    Bank_data, Rack_data = data_manipulation(list1, list2)
    PCS_data, ETC_data = list3, list4

    data = {
        "timestamp": str(seoultime),
        "Bank_data": Bank_data,
        "Rack_data": Rack_data,
        "PCS_data": PCS_data,
        "ETC_data": ETC_data,
    }

    print(data)
    producer = KafkaProducer(
        acks=0,
        compression_type=None,
        bootstrap_servers=[
            "카프카클러스터주소1",
            "카프카클러스터주소2",
            "카프카클러스터주소3",
        ],
        value_serializer=lambda x: dumps(x).encode("utf-8"),
    )

    producer.send("sionyu", value=data)
    #    producer.flust()
    producer.close()

    end = time.time()
    print(end - start)


class thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        main()


def job():
    make_thread = thread()
    make_thread.run()


if __name__ == "__main__":

    file_path = "파일경로"
    log_path = "로그경로"

    start = int(time.time())

    while True:
        try:
            main_logger = logs.get_logger("operation2", log_path, "operation2.json")
            while True:
                end = int(time.time())
                if start == end - 1:
                    job()
                time.sleep(0.5)
                start = end
        except Exception as e:
            log_massage = """all sections error : {error}""".format(error=e)
            main_logger.error(log_massage)
            time.sleep(2)
            continue
