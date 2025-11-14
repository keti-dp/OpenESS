# 베이스 이미지 설정
FROM python:3.9

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일 복사
COPY . /app

# 종속성 설치
RUN pip install --no-cache-dir psycopg2 numpy

# 환경 변수 파일 복사 (필요할 경우)
# COPY novous.env /app

# 스크립트 실행
CMD ["python", "sos_container.py"]
