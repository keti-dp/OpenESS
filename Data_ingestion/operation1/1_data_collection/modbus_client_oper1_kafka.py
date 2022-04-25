#!/bin/python

"""
modbus_client_oper1_kafka.py : 태양광 ESS 데이터 수집을 위한 코드 (카프카 연동버전)

        ---------------------------------------------------------------------------
        Copyright(C) 2021, 윤태일 / KETI / taeil777@keti.re.kr

        아파치 라이선스 버전 2.0(라이선스)에 따라 라이선스가 부여됩니다.
        라이선스를 준수하지 않는 한 이 파일을 사용할 수 없습니다.
        다음에서 라이선스 사본을 얻을 수 있습니다.

        http://www.apache.org/licenses/LICENSE-2.0

        관련 법률에서 요구하거나 서면으로 동의하지 않는 한 소프트웨어
        라이선스에 따라 배포되는 것은 '있는 그대로' 배포되며,
        명시적이든 묵시적이든 어떠한 종류의 보증이나 조건도 제공하지 않습니다.
        라이선스에 따른 권한 및 제한 사항을 관리하는 특정 언어는 라이선스를 참조하십시오.
        ---------------------------------------------------------------------------

        ---------------------------------------------------------------------------
        The MIT License

        Copyright(C) 2021, 윤태일 / KETI / taeil777@keti.re.kr

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in
        all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
        THE SOFTWARE.
        ---------------------------------------------------------------------------


최신 테스트 버전 : 1.0.0 ver
최신 안정화 버전 : 1.0.0 ver

시온유 태양광 ESS데이터 수집을 위한 코드입니다.

Kafka cluster 연동을 통해 1초 단위로 데이터가 수집되며 

파싱해온 데이터를 저장규격에 맞게 필터링하고 재가공하여

GCP 인스턴스에 구축된 Timescale DB에 저장합니다.

전체적인 코드에 대한 설명은 https://github.com/keti-dp/OpenESS 에서 확인하실 수 있습니다.
       

"""
from pyModbusTCP.client import ModbusClient

import time
import timescale_input_test
import datetime

from multiprocessing import Process
import threading
from pytz import timezone
import numpy as np
import logs
import os
import threading

from kafka import KafkaConsumer
from json import loads


def data_preprocessing(partID, data_dict):
    """데이터 전처리 코드"""

    if partID == 4:
        data_dict["ETC"].pop(3)
        data = data_dict["ETC"]
        scale_factor_file_path = "/scalefactor_ETC.txt"
        value_range_file_path = "/range_ETC.txt"
    # BMS1, 2의 경우 데이터 파싱 개수를 초과하기때문에 나눠서 처리
    elif partID == 1:
        data = data_dict["BMS1-1"] + data_dict["BMS1-2"]
        scale_factor_file_path = "/scalefactor_BMS1.txt"
        value_range_file_path = "/range_BMS1.txt"

    elif partID == 2:
        data = data_dict["BMS2-1"] + data_dict["BMS2-2"] + data_dict["BMS2-3"]
        scale_factor_file_path = "/scalefactor_BMS2.txt"
        # BMS2는 0 1값밖에 없기 때문에 인트형으로 변경
        for i in range(len(data)):
            data[i] = int(data[i])
    else:
        data = data_dict["PCS"]
        scale_factor_file_path = "/scalefactor_PCS.txt"

    # 뱅크 아이디 추가
    if partID == 3 or partID == 4:
        BANK_ID = 1
        data.insert(0, BANK_ID)

    preprocessing_logger = logs.get_logger("operation1", "./log/", "operation1.json")

    try:

        # 스케일 팩터파일
        scale_factor_file = open(scale_factor_file_path, "r")
        scale_factor = scale_factor_file.readlines()

        # 파트 2, 3
        if partID == 2:
            for i in range(len(data)):
                if in_range(data[i], 0, 3):
                    pass
                else:
                    print("partID : ", partID)
                    print("번호 : ", i)
                    print("범위를 벗어난 값 입니다.", data[i])
                    preprocessing_logger.warning(
                        "An out-of-range value occurs at address %s of part %s : %s",
                        i,
                        partID,
                        data[i],
                    )
            scale_factor_file.close()
            return data

        elif partID == 3:
            pass
        else:
            value_range_file = open(value_range_file_path, "r")
            value_range = value_range_file.readlines()

        # 값 범위(레인지) 판별
        # PCS의 경우 값 범위가 없기때문에 그냥 리턴
        if partID == 3:
            scale_factor_file.close()
            # preprocessing_logger.info("part %s data preprocessing success", partID)
            return data

        # 스케일 팩터 적용
        for i in range(len(data)):
            scale_factor[i] = scale_factor[i].strip("\n")  # 스케일팩터 줄바꿈 문자 제거
            data[i] = int(data[i]) * float(scale_factor[i])  # 스케일팩터 적용
            data[i] = float("{:.3f}".format(data[i]))  # 소수점 한자리 적용

        for i in range(len(data)):
            value_range[i] = value_range[i].strip("\n")  # 레인지 줄바꿈 문자 제거
            min_value = float(value_range[i].split()[0])
            max_value = float(value_range[i].split()[2])

            if in_range(data[i], min_value, max_value):
                pass
            else:
                print("partID : ", partID)
                print("번호 : ", i)
                print("범위를 벗어난 값 입니다.", data[i])
                preprocessing_logger.warning(
                    "An out-of-range value occurs at address %s of part %s : %s",
                    i,
                    partID,
                    data[i],
                )

        scale_factor_file.close()
        value_range_file.close()

        # preprocessing_logger.info("part %s data preprocessing success", partID)

        return data
    except Exception as e:
        preprocessing_logger.error("part %s data preprocessing error : ", partID, e)


# 최대값 최소값 사이에 존재하는지 판별
def in_range(value, min, max):
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


if __name__ == "__main__":

    while True:
        operation_site = "operation1"
        topic_name = "operation1"
        consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=[
                "kafka cluster1",
                "kafka cluster2",
                "kafka cluster3",
            ],
            auto_offset_reset="latest",
            enable_auto_commit=True,
            # group_id=None,
            group_id="gcp_oper1",
            value_deserializer=lambda x: loads(x.decode("utf-8")),
            consumer_timeout_ms=1000,
        )
        print("[begin] get consumer list")

        for message in consumer:

            print(message)

            list1 = data_preprocessing(1, message.value["fields"])
            list2 = data_preprocessing(2, message.value["fields"])
            list3 = data_preprocessing(3, message.value["fields"])
            list4 = data_preprocessing(4, message.value["fields"])

            seoultime = datetime.datetime.fromtimestamp(
                message.timestamp / 1000
            ).replace(microsecond=0)

            print(seoultime)

            Bank_data, Rack_data = data_manipulation(list1, list2)
            PCS_data, ETC_data = list3, list4

            timescale = timescale_input_test.timescale()
            timescale.Bank_input_data(seoultime, Bank_data, operation_site)
            timescale.Rack_input_data(seoultime, Rack_data, operation_site)
            timescale.PCS_input_data(seoultime, PCS_data, operation_site)
            timescale.ETC_input_data(seoultime, ETC_data, operation_site)

            print("[end] get consumer list")

        time.sleep(1)
