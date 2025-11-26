"""
sos_container.py : TAG_SET 기반 실시간/준실시간 통합 안전지표(SOS) 계산용 Docker 컨테이너 실행 코드

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

환경변수(DB_HOST, DB_NAME, SET_ID, TAG_SET 등)로 전달된 설정을 기반으로 ESS 운영 데이터베이스에서
각 태그의 최신 값을 주기적으로 조회하고, 임계치 기반 멤버십 함수 계산을 통해 통합 안전지표(SOS)를
산출한 뒤 sos 테이블에 적재하기 위한 컨테이너 실행용 코드입니다.

multiprocessing 및 세마포어를 활용하여 주기적 스케줄링과 동시 실행 개수 제어를 수행하며,
SET_ID별 통합 메시지(INTEGRATED_LOG_MESSAGES)를 통해 안전/경고/위험 상태 메시지를 생성하는
실시간(또는 준실시간) 안전지표 계산 모듈로 활용함.

[외부 오픈소스 라이브러리 및 라이선스 안내]
이 파일은 다음과 같은 외부 오픈소스 라이브러리를 사용함.
  - psycopg2   : LGPL License (with exceptions)
  - NumPy      : BSD-3-Clause License

각 외부 라이브러리의 상세 라이선스 전문은 해당 프로젝트의 LICENSE 파일을 참조함.

"""

import os
import psycopg2
import json
import datetime
import time
import math
from multiprocessing import Process, Semaphore
from zoneinfo import ZoneInfo
import numpy as np

INTEGRATED_LOG_MESSAGES = {
    "1": {
        "safe": {"level": "안전", "msg": "SET1 정상가동"},
        "warn": {"level": "경고", "msg": "SET1 편차 감지됨. 점검 필요"},
        "danger": {"level": "위험", "msg": "SET1 이상 감지됨. 즉시 점검 요망"},
    },
    "2": {
        "safe": {"level": "안전", "msg": "SET2 정상가동"},
        "warn": {
            "level": "경고",
            "msg": "Rdiff/Cdiff 편차 감지됨. ESS 점검 및 조기 유지보수 필요",
        },
        "danger": {
            "level": "위험",
            "msg": "이상 셀 감지됨. ESS 사용 중단 및 즉시 현장 점검 요망",
        },
    },
    # 1: { ... } 필요하다면 int 키도 추가 가능!
}

R_WARN, R_UNSAFE = 0.021, 0.040
C_WARN, C_UNSAFE = 0.010, 0.020

# --- 잔여수명 캘리브레이션(선형보간)
Y20, Y100 = 5.0, 25.0  # 20사이클=5년, 100사이클=25년
R_AT20, R_AT100 = 0, 0.04  # Rdiff dev 기준점
C_AT20, C_AT100 = 0, 0.02  # Cdiff dev 기준점
Y_MAX = Y100

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
set_id = os.getenv("SET_ID")  # 보통 str, 환경변수면!

tag_set_str = os.getenv("TAG_SET")
if tag_set_str:
    tag_set = json.loads(tag_set_str)
else:
    print("TAG_SET 환경 변수가 설정되지 않았습니다.")
    tag_set = {}


# 잔여수명: dev -> 사용연한(년)을 선형보간 후 잔여수명 = Y_MAX - 사용연한
def life_from_dev(dev, d20, d100, y20=Y20, y100=Y100):
    if not np.isfinite(dev):
        return np.nan
    # dev가 범위 밖이면 외삽 허용 → 마지막에 [0, Y_MAX]로 클램프
    if d100 == d20:
        y = y100
    else:
        ratio = (dev - d20) / (d100 - d20)
        y = y20 + ratio * (y100 - y20)  # 사용연한(년) 추정
    life_left = max(0.0, Y_MAX - y)
    return float(life_left)


def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )


def get_last_sos_timestamp_for_set(conn, set_id):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT "TIMESTAMP"
        FROM sos
        WHERE "SET_ID" = %s
        ORDER BY "TIMESTAMP" DESC
        LIMIT 1
        """,
        (set_id,),
    )
    row = cur.fetchone()
    cur.close()
    if row:
        return row[0]
    return None


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


def process_data():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        f_safety_dict = {}
        log_messages = []
        max_data_ts = None

        for tag, thresholds in tag_set.items():
            cursor.execute(
                """
                SELECT "TABLE_NAME", "WARN_ACTION", "UNSAFE_ACTION"
                FROM tag_list
                WHERE "TAG" = %s
                """,
                (tag,),
            )
            table_info = cursor.fetchone()
            if not table_info:
                print(f"[WARN] tag_list에 '{tag}'가 없습니다.")
                continue
            table_name, warn_action, unsafe_action = table_info

            cursor.execute(
                f"""
                SELECT "TIMESTAMP", "VALUE"
                FROM {table_name}
                WHERE "TAG" = %s
                ORDER BY "TIMESTAMP" DESC
                LIMIT 1
                """,
                (tag,),
            )
            data_row = cursor.fetchone()
            if not data_row:
                print(
                    f"[WARN] '{table_name}' 테이블에서 TAG='{tag}' 데이터가 없습니다."
                )
                continue

            data_ts, value = data_row

            if (max_data_ts is None) or (data_ts > max_data_ts):
                max_data_ts = data_ts

            warn_threshold = thresholds["WARN_THRESHOLD"]
            unwarn_threshold = thresholds["UNSAFE_THRESHOLD"]
            f_safety_dict[tag], w_val = adaptive_sigmoid(
                value, warn_threshold, unwarn_threshold, 0.95
            )

            if warn_threshold < unwarn_threshold:
                if value > unwarn_threshold:
                    level = "위험"
                    message = unsafe_action if unsafe_action else "위험: 임계치 초과"
                elif value > warn_threshold:
                    level = "경고"
                    message = warn_action if warn_action else "경고: 임계치 초과"
                else:
                    level = "안전"
                    message = "정상"
            else:
                if value < unwarn_threshold:
                    level = "위험"
                    message = unsafe_action if unsafe_action else "위험: 임계치 초과"
                elif value < warn_threshold:
                    level = "경고"
                    message = warn_action if warn_action else "경고: 임계치 초과"
                else:
                    level = "안전"
                    message = "정상"

            log_messages.append(
                {
                    "tag": tag,
                    "timestamp": str(data_ts),
                    "value": value,
                    "level": level,
                    "message": message,  # <- 여기도 key명 message로 수정
                }
            )

        if f_safety_dict:
            sos = 1.0
            for val in f_safety_dict.values():
                sos *= val
        else:
            sos = 1.0

        if max_data_ts is None:
            print("[INFO] 처리할 데이터가 없습니다. 종료.")
            return

        # ---- set_id별 통합 메시지 적용 (KeyError 절대 안남!) ----
        msg_dict = None
        # 우선 str(set_id)
        # ------ 메시지 딕셔너리 참조 (KeyError 방지) ------
        msg_dict = INTEGRATED_LOG_MESSAGES.get(set_id)
        if msg_dict is None:
            msg_dict = INTEGRATED_LOG_MESSAGES.get(1)
        if msg_dict is None:
            # 1번도 없으면 아래 기본값으로
            msg_dict = {
                "safe": {"level": "안전", "msg": "안전메시지"},
                "warn": {"level": "경고", "msg": "경고메시지"},
                "danger": {"level": "위험", "msg": "위험메시지"},
            }

        # taeil
        # Rdiff_remaining_life = life_from_dev(
        #     f_safety_dict["Rdiff_dev_20"], R_AT20, R_AT100, Y20, Y100
        # )
        # Cdiff_remaining_life = life_from_dev(
        #     f_safety_dict["Cdiff_dev_20"], R_AT20, R_AT100, Y20, Y100
        # )

        # min_life = min(Rdiff_remaining_life, Cdiff_remaining_life)
        # life_msg = f"잔여수명≈{min_life:.2f}년, (R:{Rdiff_remaining_life:.2f}년, C:{Cdiff_remaining_life:.2f}년)"

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

        log_messages.append(
            {
                "tag": "integrated_log",
                "timestamp": str(max_data_ts),
                "level": level,
                "msg": msg,
            }
        )
        # --------------------------------------------

        message_str = (
            json.dumps(log_messages, ensure_ascii=False) if log_messages else None
        )

        cursor.execute("SET TIMEZONE TO 'Asia/Seoul';")
        cursor.execute(
            """
            INSERT INTO sos ("TIMESTAMP", "VALUE", "SET_ID", "F_VALUE", "MESSAGE")
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                max_data_ts,
                round(float(sos), 3),  # numpy → Python float 변환
                set_id,
                json.dumps(f_safety_dict, ensure_ascii=False),
                message_str,
            ),
        )
        conn.commit()

        cursor.close()
        conn.close()

        print(
            f"[INFO] process_data() 완료 - new row inserted with TS={max_data_ts}, SOS={sos}"
        )

    except Exception as e:
        print(f"[ERROR] process_data() 실패: {e}")


def run_task(semaphore):
    try:
        process_data()
    finally:
        semaphore.release()


def schedule_task():
    max_processes = 5
    semaphore = Semaphore(max_processes)
    seoul_tz = ZoneInfo("Asia/Seoul")

    while True:
        start_time = datetime.datetime.now(seoul_tz)

        conn = get_db_connection()
        last_sos_ts = get_last_sos_timestamp_for_set(conn, set_id)
        if not last_sos_ts:
            last_sos_ts = datetime.datetime(1970, 1, 1, tzinfo=seoul_tz)

        new_data_found = False

        cur = conn.cursor()
        for tag in tag_set.keys():
            cur.execute(
                """
                SELECT "TABLE_NAME"
                FROM tag_list
                WHERE "TAG" = %s
                """,
                (tag,),
            )
            res = cur.fetchone()
            if not res:
                continue
            table_name = res[0]

            cur.execute(
                f"""
                SELECT "TIMESTAMP"
                FROM {table_name}
                WHERE "TAG" = %s
                ORDER BY "TIMESTAMP" DESC
                LIMIT 1
                """,
                (tag,),
            )
            row = cur.fetchone()
            if not row:
                continue

            current_latest_ts = row[0]
            if current_latest_ts > last_sos_ts:
                new_data_found = True
                break

        cur.close()
        conn.close()

        if new_data_found:
            print("[INFO] New data found. Starting process_data() ...")
            semaphore.acquire()
            p = Process(target=run_task, args=(semaphore,))
            p.start()
        else:
            print("[INFO] No new data. Skip this round.")

        elapsed = (datetime.datetime.now(seoul_tz) - start_time).total_seconds()
        time.sleep(max(1 - elapsed, 0))


if __name__ == "__main__":
    schedule_task()
