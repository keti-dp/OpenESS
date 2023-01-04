#!/bin/python


"""
kafka_oper2_bank1.py : 태양광 ESS 데이터 수집을 위한 코드 (카프카 producer)

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

판리 태양광 ESS데이터 수집을 위한 코드입니다.

구축된 kafka cluster에서 producer를 생성하여 modbus TCP통신을 통해 ESS 운영사이트에서 데이터를 저장합니다.

전체적인 코드에 대한 설명은 https://github.com/keti-dp/OpenESS 에서 확인하실 수 있습니다.
       

"""


from pyModbusTCP.client import ModbusClient
import json
from json import dumps
import time

from datetime import datetime
from multiprocessing import Process
import threading
from pytz import timezone
import numpy as np
import logs
import pprint
from kafka import KafkaProducer


class ESS_Modbus:

    # 기본 클라이언트 설정
    def __init__(self):
        try:
            connection_logger = logs.get_logger("operation2", log_path, "operation2.json")
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
        """인셀 EMS 프로토콜 기준으로 데이터를 받아오는 메소드

        Args:
            partID (int): 인셀 EMS 프로토콜 기준 BMS1, BMS2, PCS, ETC 중 받아올 데이터에 대한 숫자
            BMS1, 2의 경우 속도적인 문제로 bank1과 bank2를 구분하여 파싱함

        Returns:
            list: 인셀 EMS 프로토콜 기준으로 데이터가 담겨있음

        Example:
            >>> BMS1_data = data_parsing(12)
            >>> BMS2_data = data_parsing(21)
            >>> PCS_data = data_parsing(3)
            >>> ETC_data = data_parsing(4)
            ['0','0','1','1',...]
        """

        try:
            # 파싱로그
            parsing_logger = logs.get_logger("operation2", log_path, "operation2.json")

            # 인풋레지스터의 경우 최대 125개밖에 못가져오기때문에 BMS데이터의 경우 수정이 필요함
            if partID == 11:  # BMS1 - bank1
                start_address = 0
                # end_address = 219
            elif partID == 12:  # BMS1 - bank2
                start_address = 220
                # end_address = 416
            elif partID == 21:  # BMS2 - bank1
                start_address = 500
                # end_address = 1002
            elif partID == 22:  # BMS2 - bank2
                start_address = 1003
                # end_address = 1453
            elif partID == 3:  # PCS
                start_address = 1500
                # end_address = 1552
            elif partID == 4:  # ETC
                start_address = 1600
                # end_address = 1605

            # 파싱데이터 리스트
            data = []

            # 데이터 파싱

            # BMS1, 2의 경우 데이터 파싱 개수를 초과하기때문에 나눠서 처리

            if partID == 11:  # bms1 bank1에 대한 데이터
                data1 = self.client.read_input_registers(start_address, 125)  # 0~124
                data2 = self.client.read_input_registers(125, 95)  # 125~219
                data = data1 + data2

            elif partID == 12:  # bms1 bank2에 대한 데이터
                data1 = self.client.read_input_registers(start_address, 125)
                data2 = self.client.read_input_registers(345, 72)
                data = data1 + data2

            elif partID == 21:  # bms2 bank1에 대한 데이터
                data1 = self.client.read_input_registers(start_address, 125)
                data2 = self.client.read_input_registers(625, 125)
                data3 = self.client.read_input_registers(750, 125)
                data4 = self.client.read_input_registers(875, 125)
                data5 = self.client.read_input_registers(1000, 3)

                data = data1 + data2 + data3 + data4 + data5

                # print("BMS2 bank1", len(data))

            elif partID == 22:  # bms2 bank1에 대한 데이터
                data1 = self.client.read_input_registers(start_address, 125)
                data2 = self.client.read_input_registers(1128, 125)
                data3 = self.client.read_input_registers(1253, 125)
                data4 = self.client.read_input_registers(1378, 76)
                data = data1 + data2 + data3 + data4

            elif partID == 3:
                data1 = self.client.read_input_registers(start_address, 53)
                data = data1

            elif partID == 4:
                data1 = self.client.read_input_registers(start_address, 6)
                data = data1

            # parsing_logger.info("part %s data parsing success", partID)

        except Exception as e:
            log_massage = """part {partID} data parsing error : {error}""".format(
                partID=partID, error=e
            )
            parsing_logger.error(log_massage)

        # TO-DO

        # 데이터 전처리
        result = self.data_preprocessing(partID, data)

        # 뱅크 아이디 추가
        # if partID == 3 or partID == 4:
        #     BANK_ID = 1
        #     data.insert(0, BANK_ID)
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
        """modbusTCP 통신을 통해 받아온 데이터를 전처리하는(scale factor, 음수변환, 값 범위확인) 메소드

        Args:
            partID (int): 인셀 EMS 프로토콜 기준 BMS1, BMS2, PCS, ETC 중 전처리할 데이터에 대한 숫자
            BMS1, 2의 경우 속도적인 문제로 bank1과 bank2를 구분하여 파싱함
            data (list): modbusTCP통신으로 받아온 원천데이터

        Returns:
            list: 전처리작업이 완료된 데이터

        Example:
            >>> BMS1_data = data_parsing(12, BMS)
            >>> BMS1_preprocessing_data = data_preprocessing(12, BMS1_data)
            ['0','0','1','1',...]
        """

        preprocessing_logger = logs.get_logger("operation2", log_path, "operation2.json")

        try:
            if partID == 11:  # BMS1 - bank1
                scale_factor_file_path = file_path + "scalefactor_BMS1_bank1.txt"

            elif partID == 12:  # BMS1 - bank2
                scale_factor_file_path = file_path + "scalefactor_BMS1_bank2.txt"

            # BMS2는 0 1값밖에 없기 때문에 필요없음
            elif partID == 3:  # PCS
                scale_factor_file_path = file_path + "scalefactor_PCS.txt"

            elif partID == 4:  # ETC
                scale_factor_file_path = file_path + "scalefactor_ETC.txt"

            if partID != 21 and partID != 22:
                # 스케일 팩터파일
                scale_factor_file = open(scale_factor_file_path, "r")
                scale_factor = scale_factor_file.readlines()

            # 음수 변환
            for i in range(len(data)):
                if data[i] <= 32767:
                    pass
                elif 32767 < data[i] < 65536:
                    data[i] = -(65535 - data[i] + 1)

            # part2의 값 범위와 scale_factor 적용
            # if partID == 21 or 22:
            # for i in range(len(data)):
            #     if self.in_range(data[i], 0, 3):
            #         pass
            #     else:
            #         print("partID : ", partID)
            #         print("번호 : ", i)
            #         print("범위를 벗어난 값 입니다.", data[i])
            #         preprocessing_logger.warning(
            #             "An out-of-range value occurs at address %s of part %s : %s",
            #             i,
            #             partID,
            #             data[i],
            #         )
            # scale_factor_file.close()
            # return data
            # elif partID == 3:
            #     pass
            # else:
            #     value_range_file = open(value_range_file_path, "r")
            #     value_range = value_range_file.readlines()

            # 스케일 팩터 적용 BMS2 는 적용시킬필요가없음
            if partID != 21 and partID != 22:
                for i in range(len(data)):
                    scale_factor[i] = scale_factor[i].strip("\n")  # 스케일팩터 줄바꿈 문자 제거
                    data[i] = int(data[i]) * float(scale_factor[i])  # 스케일팩터 적용
                    data[i] = float("{:.3f}".format(data[i]))  # 소수점 한자리 적용

            # 값 범위(레인지) 판별
            # PCS의 경우 값 범위가 없기때문에 그냥 리턴
            # if partID == 3:
            #     scale_factor_file.close()
            #     # preprocessing_logger.info("part %s data preprocessing success", partID)
            #     return data

            # for i in range(len(data)):
            #     value_range[i] = value_range[i].strip("\n")  # 레인지 줄바꿈 문자 제거
            #     min_value = float(value_range[i].split()[0])
            #     max_value = float(value_range[i].split()[2])

            #     if self.in_range(data[i], min_value, max_value):
            #         pass
            #     else:
            #         print("partID : ", partID)
            #         print("번호 : ", i)
            #         print("범위를 벗어난 값 입니다.", data[i])
            #         preprocessing_logger.warning(
            #             "An out-of-range value occurs at address %s of part %s : %s",
            #             i,
            #             partID,
            #             data[i],
            #         )
            if partID != 21 and partID != 22:
                # 스케일 팩터파일
                scale_factor_file.close()

            # value_range_file.close()

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
def data_manipulation(BMS1, BMS2, bank_id):
    """BMS1, 2 데이터를 조작하여 DB상에 넣기위한 메소드 bank데이터와 rack데이터를 만든다.

    Args:
        BMS1 (list): 전처리가 끝난 BMS1 데이터
        BMS2 (list): 전처리가 끝난 BMS2 데이터
        bankid (int): bank 번호

    Returns:
        list: BANK 데이터
        list: RACK 데이터

    Example:
        >>> Bank, Rack = data_manipulation(bms1_list, bms2_list)
    """

    # BANK 데이터 만들기
    # bank_id에 따라 Rack 개수결정
    BANK_ID = bank_id  # bank_id에 따라서 rack가 수가 달라 코드가 다름

    if BANK_ID == 1:
        info_num = 44
        rack_num = 9
    else:
        info_num = 43
        rack_num = 8

    BMS1_bank = BMS1[:13]
    BMS1_rack = BMS1[13:]
    BMS2_bank = BMS2[:info_num]
    BMS2_rack = BMS2[info_num:]

    # print("BMS1_rack : ", BMS1_rack)
    # print("BMS1_rack : ", len(BMS1_rack))

    # print("BMS2_rack : ", BMS2_rack)
    # print("BMS2_rack : ", len(BMS2_rack))

    # 뱅크 부분 0 1 반전 있음 536번
    # 541 Battery Status for Run 0, 1 반전
    BMS2_bank[-8] = 1 if BMS2_bank[-8] == 0 else 0
    BMS2_bank[-3] = 1 if BMS2_bank[-3] == 0 else 0

    # bank id 삽입
    BMS1_bank.insert(0, BANK_ID)

    # 0:정상 1:이상으로 치환 rack_num

    for i in range(rack_num):
        if BMS2_bank[22 + i] == 0:
            BMS2_bank[22 + i] = 1
        elif BMS2_bank[22 + i] == 1:
            BMS2_bank[22 + i] = 0

    # Rack status for run 에 대한게 수정 필요

    # 랙 부분 0 1 반전 있음 544 545 549

    for i in range(rack_num):
        if BMS2_rack[51 * i] == 0:
            BMS2_rack[51 * i] = 1
        elif BMS2_rack[51 * i] == 1:
            BMS2_rack[51 * i] = 0

        if BMS2_rack[1 + 51 * i] == 0:
            BMS2_rack[1 + 51 * i] = 1
        elif BMS2_rack[1 + 51 * i] == 1:
            BMS2_rack[1 + 51 * i] = 0

        if BMS2_rack[5 + 51 * i] == 0:
            BMS2_rack[5 + 51 * i] = 1
        elif BMS2_rack[5 + 51 * i] == 1:
            BMS2_rack[5 + 51 * i] = 0

    # print("BANK : ", BANK)

    # RACK 데이터 만들기 이중리스트
    bank_commuication_fault_dict = {}  # bank json
    Rack_list = []
    for rack_number in range(1, rack_num + 1):
        temp1 = BMS1_rack[23 * (rack_number - 1) : 23 * rack_number]  # 자르기
        temp2 = BMS2_rack[51 * (rack_number - 1) : 51 * rack_number]

        # print("temp1 : ", temp1)
        # print("temp2 : ", temp2)

        rack_module_fault_dict = {}  # rack json

        for module_number in range(1, 21):
            rack_module_fault_dict["module" + str(module_number)] = temp2[-20 + module_number - 1]

        # json 대체
        temp2[-20] = json.dumps(rack_module_fault_dict)
        temp2_2 = temp2[0:-19]

        bank_commuication_fault_dict["rack" + str(rack_number)] = BMS2_bank[22 + rack_number - 1]

        temp3 = temp1 + temp2_2
        temp3.insert(0, rack_number)  # BANK, RACK ID 추가
        temp3.insert(0, BANK_ID)  # BANK, RACK ID 추가
        Rack_list.append(temp3)

    # json 파일에 담기 dict로 만들어서 json 으로 변환

    BMS2_bank[22] = json.dumps(bank_commuication_fault_dict)

    # 변환
    for i in range(rack_num - 1):
        del BMS2_bank[23]

    BANK = BMS1_bank + BMS2_bank

    return BANK, Rack_list


def main():

    seoultime = datetime.now(timezone("asia/seoul")).replace(microsecond=0)

    # 운영사이트
    operation_site = "operation2"

    print(seoultime)

    start = time.time()

    test1 = ESS_Modbus()

    list1 = test1.data_parsing(11)
    list2 = test1.data_parsing(21)
    # list3 = test1.data_parsing(12)
    # list4 = test1.data_parsing(22)
    # list5 = test1.data_parsing(3)
    # list6 = test1.data_parsing(4)

    Bank_data, Rack_data = data_manipulation(list1, list2, bank_id=1)

    # for i in range(len(Bank_data)):
    #     print(i, Bank_data[i])

    # for i in range(len(Rack_data)):
    #     print(i, Rack_data[i])

    # Bank_data2, Rack_data2 = data_manipulation(list3, list4, bank_id=2)

    # PCS_data, ETC_data = list5, list6

    Bank_data_list = []
    Bank_data_list.append(Bank_data)
    # Bank_data_list.append(Bank_data2)

    Rack_data_list = []
    Rack_data_list.append(Rack_data)
    # Rack_data_list.append(Rack_data2)

    data = {
        "timestamp": str(seoultime),
        "Bank_data": Bank_data_list,
        "Rack_data": Rack_data_list,
        # "PCS_data": PCS_data,
        # "ETC_data": ETC_data,
    }

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

    producer.send("panly", value=data)
    #    producer.flust()
    producer.close()

    # timescale = timescale_input_test.timescale(
    #     ip="34.64.147.133",
    #     port="5432",
    #     username="postgres",
    #     password="keti1234!",
    #     dbname="ESS_Operating_Site2",
    # )

    # timescale.Bank_input_data(seoultime, Bank_data, operation_site)
    # timescale.Rack_input_data(seoultime, Rack_data, operation_site)
    # timescale.Bank_input_data(seoultime, Bank_data2, operation_site)
    # timescale.Rack_input_data(seoultime, Rack_data2, operation_site)
    # timescale.PCS_input_data(seoultime, PCS_data, operation_site)
    # timescale.ETC_input_data(seoultime, ETC_data, operation_site)

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
                time.sleep(0.2)
                start = end

        except Exception as e:
            log_massage = """all sections error : {error}""".format(error=e)
            main_logger.error(log_massage)
            time.sleep(2)
            continue
