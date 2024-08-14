# base 이미지
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 환경 변수
ENV CUDA_VERSION=11.8

# 디폴트 폴더 변경
WORKDIR /usr/workspace

# pip 패키지 설치
RUN apt-get -qq update 
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# 소스코드 복사
COPY ./ ./

LABEL org.opencontainers.image.description "BMS Resorce Data Anomaly Detection Using USAD, DeepAnt and LSTM_USAD Model"
