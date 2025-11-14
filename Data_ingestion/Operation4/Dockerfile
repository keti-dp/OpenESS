# 베이스 이미지 선택 (Python이 설치된 이미지를 기반으로)
FROM python:3.8

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 Python 라이브러리가 있다면 requirements.txt를 사용하여 설치
# requirements.txt 파일에 필요한 라이브러리 목록을 넣어둡니다.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Python 스크립트를 작업 디렉토리로 복사
COPY . /app

# 컨테이너가 시작될 때 실행될 명령어 지정
CMD ["python", "./modbus_client_oper4_kafka+protectionmap_for_docker.py"]
