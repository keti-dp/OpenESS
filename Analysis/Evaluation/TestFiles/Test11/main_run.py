#-*- coding: utf-8 -*-
"""

title: 안전SW프레임워크 1차년도 성능시험평가
date: 2021-10-12
writer: KETI 최우정

    11.수집 데이터 호환 및 데이터 연동 플로우 처리율
        -목표 달성률(%)

        1)데이터 로드(10%)
            - DB
            - Storage
        2)데이터 전처리(30%)
            - 이상 검출
            - 이상치 제거
            - 규격화 처리
        3) 가공데이터 저장(5%)
            - DB
            - Storage
        4) 분석(45%)
            - 통계분석
            - 기계학습
        5) 결과 반환(10%)
            - 출력 및 시각화

"""
import yaml
import time
import os

def read_setting():
    """
    yaml
    """
    with open(currentpath+'/setting.yaml', 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    currentpath = os.getcwd()
    setting = read_setting()

    #컴포넌트 순서대로 실행
    for component in setting['run']:
        dir = currentpath+"/Component/"+component
        print('run', component)

        # 외부 스크립트 실행
        exec(open(dir, 'rt', encoding='UTF-8').read())
        print('----------Finished----------\n\n')
        time.sleep(1)
