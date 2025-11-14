"""
sos_batch.py : 통합 안전지표(SOS) 배치 계산 및 sos 테이블 적재를 위한 배치 분석 코드

        ---------------------------------------------------------------------------
        Copyright(C) 2025, 윤태일 / KETI / taeil777@keti.re.kr

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

        Copyright(C) 2025, 윤태일 / KETI / taeil777@keti.re.kr

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

특정 SET_ID와 TAG_SET 구성에 대해 tag_list 및 개별 지표 테이블에서 데이터를 조회하고,
지정된 시간 구간(start_time ~ end_time)에 대해 통합 안전지표(SOS)를 배치 방식으로 계산하여
sos 테이블에 적재하기 위한 코드입니다.

태그별 경고/위험 임계치(WARN_THRESHOLD, UNSAFE_THRESHOLD)를 기반으로 적응형 시그모이드
멤버십 함수를 계산하고, 시점별 안전도 값과 메시지(LOG)를 생성함으로써 ESS의 안전 상태를
시간 축 전체에 걸쳐 분석·평가하는 데 활용함.

[외부 오픈소스 라이브러리 및 라이선스 안내]
이 파일은 다음과 같은 외부 오픈소스 라이브러리를 사용함.
  - psycopg2   : LGPL License (with exceptions)
  - NumPy      : BSD-3-Clause License
  - pandas     : BSD-3-Clause License

각 외부 라이브러리의 상세 라이선스 전문은 해당 프로젝트의 LICENSE 파일을 참조함.

"""

import os
import psycopg2
import json
import datetime
import time
import numpy as np
import math
from zoneinfo import ZoneInfo
import traceback
import pandas as pd

DB_HOST = "1.214.41.251"
DB_PORT = "5434"
DB_NAME = "PUBLIC_DB"
DB_USER = "postgres"
DB_PASSWORD = "keti1234!"

set_id = 30  # int 타입임!


tag_set = {
    "평균전압상승률": {"WARN_THRESHOLD": 0.5, "UNSAFE_THRESHOLD": 1.45},
    "HI_방전시간단축률": {"WARN_THRESHOLD": -0.3, "UNSAFE_THRESHOLD": -0.5},
}


INTEGRATED_LOG_MESSAGES = {
    30: {
        "safe": {"level": "안전", "msg": "정상가동"},
        "warn": {
            "level": "경고",
            "msg": "배터리 knee point 발생 가능성 / ESS 점검 및 배터리 SOH 확인 필요",
        },
        "danger": {
            "level": "위험",
            "msg": "배터리 노화에 따른 배터리 stress 누적 가능성 / ESS 일시 중단 및 배터리 교체 필요 ",
        },
    },
    1: {
        "safe": {"level": "안전", "msg": "기본 정상가동"},
        "warn": {"level": "경고", "msg": "기본 경고 메시지"},
        "danger": {"level": "위험", "msg": "기본 위험 메시지"},
    },
}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def calculate_w(h_warn, h_unsafe, p=0.95):
    h_median = (h_warn + h_unsafe) / 2
    return -np.log(1 / p - 1) / (h_warn - h_median)


def adaptive_sigmoid(hX: float, hX_warn: float, hX_unsafe: float, p: float):
    hX_median = 0.5 * (hX_warn + hX_unsafe)
    w = calculate_w(hX_warn, hX_unsafe, p)
    b = -w * hX_median

    if hX_warn < hX_unsafe:
        if hX < hX_warn:
            f = 1.0
        elif hX > hX_unsafe:
            f = 0.0
        else:
            z = w * hX + b
            f = sigmoid(z)
    else:
        if hX > hX_warn:
            f = 1.0
        elif hX < hX_unsafe:
            f = 0.0
        else:
            z = w * hX + b
            f = sigmoid(z)
    return f, w


def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )


def get_tag_info(conn, tag):
    """
    tag_list에서 해당 태그의 테이블명, WARN_ACTION, UNSAFE_ACTION을 가져온다.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT "TABLE_NAME", "WARN_ACTION", "UNSAFE_ACTION"
        FROM tag_list
        WHERE "TAG" = %s
        """,
        (tag,),
    )
    result = cursor.fetchone()
    cursor.close()
    if result:
        return result  # (table_name, warn_action, unsafe_action)
    else:
        return None, None, None


def get_data_for_tag(conn, tag, start_time, end_time):
    table_name, _, _ = get_tag_info(conn, tag)
    if not table_name:
        return []
    cursor = conn.cursor()
    cursor.execute("SET TIMEZONE TO 'Asia/Seoul';")
    query = f"""
        SELECT "TIMESTAMP", "VALUE"
        FROM {table_name}
        WHERE "TAG" = %s
          AND "TIMESTAMP" BETWEEN %s AND %s
        ORDER BY "TIMESTAMP" ASC
    """
    cursor.execute(query, (tag, start_time, end_time))
    rows = cursor.fetchall()
    cursor.close()
    return rows


def process_data_in_range(start_time, end_time):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 태그별 (table_name, warn_action, unsafe_action) 미리 캐싱
        tag_info_map = {}
        for tag in tag_set.keys():
            table_name, warn_action, unsafe_action = get_tag_info(conn, tag)
            tag_info_map[tag] = {
                "table_name": table_name,
                "warn_action": warn_action,
                "unsafe_action": unsafe_action,
            }

        data_map = {}
        all_timestamps = set()

        for tag in tag_set.keys():
            tag_data = get_data_for_tag(conn, tag, start_time, end_time)
            data_map[tag] = tag_data

            for row in tag_data:
                all_timestamps.add(row[0])

        sorted_timestamps = sorted(all_timestamps)
        tag_indexes = {tag: 0 for tag in tag_set.keys()}
        previous_sos_zero = False

        for current_ts in sorted_timestamps:
            f_safety_dict = {}
            value_dict = {}  # taeil 각 데이터별 dict;
            #
            log_messages = []

            # taeil

            for tag, thresholds in tag_set.items():
                warn_threshold = thresholds["WARN_THRESHOLD"]
                unsafe_threshold = thresholds["UNSAFE_THRESHOLD"]
                data_list = data_map[tag]
                tag_info = tag_info_map[tag]
                warn_action = tag_info["warn_action"]
                unsafe_action = tag_info["unsafe_action"]

                if not data_list:
                    f_safety_dict[tag] = None
                    value_dict[tag] = None
                    continue
                idx = tag_indexes[tag]
                if idx >= len(data_list):
                    value_timestamp, value = data_list[-1]

                    if value_timestamp > current_ts:
                        f_safety_dict[tag] = None
                        value_dict[tag] = None
                    else:
                        f_safety_dict[tag], w_val = adaptive_sigmoid(
                            abs(value), warn_threshold, unsafe_threshold, 0.95
                        )
                        value_dict[tag] = abs(value)

                    continue
                if data_list[idx][0] > current_ts:
                    if idx == 0:
                        f_safety_dict[tag] = None
                        value_dict[tag] = None
                    else:
                        value_timestamp, value = data_list[idx - 1]
                        if value_timestamp <= current_ts:
                            f_safety_dict[tag], w_val = adaptive_sigmoid(
                                abs(value), warn_threshold, unsafe_threshold, 0.95
                            )
                            value_dict[tag] = abs(value)

                        else:
                            f_safety_dict[tag] = None
                            value_dict[tag] = None

                        # ===== level 및 message 처리 =====
                        if warn_threshold < unsafe_threshold:
                            if value > unsafe_threshold:
                                level = "위험"
                                message = (
                                    unsafe_action
                                    if unsafe_action
                                    else "위험: 임계치 초과"
                                )
                            elif value > warn_threshold:
                                level = "경고"
                                message = (
                                    warn_action if warn_action else "경고: 임계치 초과"
                                )
                            else:
                                level = "안전"
                                message = "정상"
                        else:
                            if value < unsafe_threshold:
                                level = "위험"
                                message = (
                                    unsafe_action
                                    if unsafe_action
                                    else "위험: 임계치 초과"
                                )
                            elif value < warn_threshold:
                                level = "경고"
                                message = (
                                    warn_action if warn_action else "경고: 임계치 초과"
                                )
                            else:
                                level = "안전"
                                message = "정상"
                        log_messages.append(
                            {
                                "tag": tag,
                                "timestamp": str(current_ts),
                                "value": value,
                                "level": level,
                                "message": message,
                            }
                        )
                else:
                    while (
                        idx + 1 < len(data_list) and data_list[idx + 1][0] <= current_ts
                    ):
                        idx += 1
                    value_timestamp, value = data_list[idx]
                    f_safety_dict[tag], w_val = adaptive_sigmoid(
                        abs(value), warn_threshold, unsafe_threshold, 0.95
                    )
                    value_dict[tag] = abs(value)
                    tag_indexes[tag] = idx

                    # ===== level 및 message 처리 =====
                    if warn_threshold < unsafe_threshold:
                        if value > unsafe_threshold:
                            level = "위험"
                            message = (
                                unsafe_action if unsafe_action else "위험: 임계치 초과"
                            )
                        elif value > warn_threshold:
                            level = "경고"
                            message = (
                                warn_action if warn_action else "경고: 임계치 초과"
                            )
                        else:
                            level = "안전"
                            message = "정상"
                    else:
                        if value < unsafe_threshold:
                            level = "위험"
                            message = (
                                unsafe_action if unsafe_action else "위험: 임계치 초과"
                            )
                        elif value < warn_threshold:
                            level = "경고"
                            message = (
                                warn_action if warn_action else "경고: 임계치 초과"
                            )
                        else:
                            level = "안전"
                            message = "정상"
                    log_messages.append(
                        {
                            "tag": tag,
                            "timestamp": str(current_ts),
                            "value": value,
                            "level": level,
                            "message": message,
                        }
                    )

            if any(v is None for v in f_safety_dict.values()):
                continue

            sos = 1.0
            for val in f_safety_dict.values():
                sos *= val

            if previous_sos_zero:
                sos = 0.0
            if sos == 0.0:
                previous_sos_zero = True

            # ------ 메시지 딕셔너리 참조 (KeyError 방지) ------
            msg_dict = INTEGRATED_LOG_MESSAGES.get(set_id)
            if msg_dict is None:
                msg_dict = INTEGRATED_LOG_MESSAGES.get(1)
            if msg_dict is None:
                msg_dict = {
                    "safe": {"level": "알수없음", "msg": "기본 메시지 없음"},
                    "warn": {"level": "알수없음", "msg": "기본 메시지 없음"},
                    "danger": {"level": "알수없음", "msg": "기본 메시지 없음"},
                }

            # ------------------------
            # msg_dict 수정 부분
            # ------------------------

            # msg_dict["warn"]["msg"] = life_msg
            # msg_dict["danger"]["msg"] = life_msg

            if sos == 1.0:
                level = msg_dict["safe"]["level"]
                msg = msg_dict["safe"]["msg"]
            elif 0 < sos < 1:
                level = msg_dict["warn"]["level"]
                msg = msg_dict["warn"]["msg"]
            else:
                level = msg_dict["danger"]["level"]
                msg = msg_dict["danger"]["msg"]

            if sos == 1.0:
                level = msg_dict["safe"]["level"]
                msg = msg_dict["safe"]["msg"]
            elif 0 < sos < 1:
                level = msg_dict["warn"]["level"]
                msg = msg_dict["warn"]["msg"]
            else:
                level = msg_dict["danger"]["level"]
                msg = msg_dict["danger"]["msg"]

            log_messages.append(
                {
                    "tag": "integrated_log",
                    "timestamp": str(current_ts),
                    "level": level,
                    "msg": msg,
                }
            )

            message_value = None
            if log_messages:
                message_value = json.dumps(log_messages, ensure_ascii=False)

            cursor.execute("SET TIMEZONE TO 'Asia/Seoul';")
            cursor.execute(
                """
                INSERT INTO sos ("TIMESTAMP", "VALUE", "SET_ID", "F_VALUE", "MESSAGE")
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    current_ts,
                    round(sos, 4),
                    set_id,
                    json.dumps(f_safety_dict, ensure_ascii=False),
                    message_value,
                ),
            )
            conn.commit()

        cursor.close()
        conn.close()
        print("모든 데이터 시점에 대한 SOS 계산 및 저장이 완료되었습니다.")

    except Exception as e:
        print(f"오류 발생: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    seoul_tz = ZoneInfo("Asia/Seoul")

    start_str = "2024-11-12 00:00:00"
    end_str = "2024-11-15 00:00:00"

    start_time = datetime.datetime.fromisoformat(start_str).replace(tzinfo=seoul_tz)
    end_time = datetime.datetime.fromisoformat(end_str).replace(tzinfo=seoul_tz)

    process_data_in_range(start_time, end_time)
