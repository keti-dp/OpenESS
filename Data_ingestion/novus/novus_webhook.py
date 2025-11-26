"""
novus_webhook.py : 통합 안전지표(TAG_SET) 웹훅 수신 및 Docker 컨테이너 실행 트리거를 위한 웹훅 서버 코드

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

Flask 기반 웹훅 서버로부터 TAG_SET 정보를 입력받아 PostgreSQL의 tag_combination_map에
SET_ID를 생성·조회하고, 신규 TAG_SET에 대해서는 Docker 컨테이너(sos_container 이미지)를
실행하여 통합 안전지표(SOS) 계산 작업을 트리거하기 위한 코드입니다.

웹훅 요청 검증, SET_ID 관리, Docker 컨테이너 실행 상태 폴링 및 결과 반환까지의
엔드포인트 로직을 포함하며, ESS 통합 안전지표 분석 파이프라인의 진입점 역할을 합니다.

[외부 오픈소스 라이브러리 및 라이선스 안내]
이 파일은 다음과 같은 외부 오픈소스 라이브러리를 사용함.
  - Flask      : BSD-3-Clause License
  - docker     : Apache License 2.0
  - psycopg2   : LGPL License (with exceptions)

각 외부 라이브러리의 상세 라이선스 전문은 해당 프로젝트의 LICENSE 파일을 참조함.

"""

from flask import Flask, request, jsonify
import psycopg2
import docker
import json
import os
from psycopg2.extras import Json
import traceback
import logging
from concurrent.futures import ThreadPoolExecutor

import time
from docker.errors import DockerException


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ThreadPoolExecutor 초기화 (필요에 따라 max_workers 조정)
executor = ThreadPoolExecutor(max_workers=10)

# PostgreSQL 연결 설정
DB_NAME = ""
USER = ""
PASSWORD = ""
HOST = ""
PORT = ""


def get_or_create_set_id_and_run_container(tag_set):
    try:
        # 데이터베이스 연결
        conn = psycopg2.connect(
            dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT
        )
        cursor = conn.cursor()

        tag_set_json = Json(tag_set)

        # SET_ID 조회
        cursor.execute(
            """
            SELECT "SET_ID"
              FROM tag_combination_map
             WHERE "TAG_SET" = %s::jsonb
        """,
            (tag_set_json,),
        )
        row = cursor.fetchone()

        if row:
            set_id = row[0]
            logger.info(f"기존 SET_ID 반환: {set_id}")
            run_container_flag = False
        else:
            # INSERT & SET_ID 생성
            cursor.execute(
                """
                INSERT INTO tag_combination_map ("TAG_SET")
                VALUES (%s)
                RETURNING "SET_ID"
            """,
                (tag_set_json,),
            )
            row = cursor.fetchone()
            if not row:
                conn.rollback()
                raise Exception("데이터 삽입에 실패하였습니다.")
            set_id = row[0]
            conn.commit()
            logger.info(f"새로운 SET_ID 생성 및 반환: {set_id}")
            run_container_flag = True

        cursor.close()
        conn.close()

        if run_container_flag:
            image = "sos_container:latest"
            container_name = f"integrated_sos_{set_id}"
            environment = {
                "TAG_SET": json.dumps(tag_set),
                "SET_ID": str(set_id),
            }

            # 동기 호출: 컨테이너 생성 완료 후 리턴값을 받음
            container_resp = run_container(image, container_name, environment)
            # 결과에 set_id 추가
            container_resp["set_id"] = set_id
            return container_resp, set_id
        else:
            return {
                "status": "skipped",
                "message": "TAG_SET already exists.",
                "set_id": set_id,
            }, set_id

    except Exception as e:
        logger.error(f"오류 발생: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}, None


import time
from docker.errors import DockerException


def run_container(image, container_name, environment, timeout=60, poll_interval=1):
    """
    - image: Docker 이미지
    - container_name: 컨테이너 이름
    - environment: 환경 변수 dict
    - timeout: 최대 대기 시간(초)
    - poll_interval: 상태 체크 간격(초)
    """
    try:
        docker_client = docker.from_env()

        # .env 파일 읽기 (필요시)
        env_file_path = "/home/keti/novus/novus.env"
        if os.path.exists(env_file_path):
            with open(env_file_path) as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        environment.setdefault(k, v)

        # 컨테이너 띄우기 (detach=True -> 바로 리턴)
        container = docker_client.containers.run(
            image=image,
            name=container_name,
            environment=environment,
            detach=True,
        )
        logger.info(f"Created container {container.id}, waiting for running state…")

        # 상태 폴링
        start = time.time()
        while True:
            # reload() 하면 container.status가 갱신됩니다
            container.reload()
            status = container.status  # e.g. 'created', 'running', 'exited', ...
            logger.debug(f"Container {container.id} status: {status}")
            if status == "running":
                logger.info(f"Container {container.id} is now running")
                return {
                    "status": "success",
                    "container_id": container.id,
                    "message": "컨테이너가 정상 실행 중입니다.",
                }
            if time.time() - start > timeout:
                # 타임아웃: 원하는 시간 안에 running이 되지 않음
                logger.error(
                    f"Container {container.id} failed to reach running state in {timeout}s"
                )
                return {
                    "status": "error",
                    "message": f"Timeout: container did not start running within {timeout}s",
                }
            time.sleep(poll_interval)

    except DockerException as e:
        logger.error(f"Docker error: {e}")
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in run_container: {e}")
        return {"status": "error", "message": str(e)}


# 웹훅 처리 엔드포인트
@app.route("/webhook", methods=["POST"])
def handle_webhook():
    # 웹훅으로 받은 JSON 데이터
    data = request.json

    # TAG_SET 파라미터 가져오기
    tag_set = data.get("tag_set")  # 요청에서 받는 TAG_SET 값 (리스트 형태)

    if not tag_set:
        return jsonify({"status": "error", "message": "TAG_SET not provided"}), 400

    # TAG_SET에 해당하는 SET_ID 가져오거나 없으면 생성하고, 없는 경우에만 컨테이너 실행
    result, set_id = get_or_create_set_id_and_run_container(tag_set)

    if set_id is None:
        return (
            jsonify({"status": "error", "message": "DB error"}),
            500,
        )

    # 결과 반환, SET_ID 포함
    result["set_id"] = set_id
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
