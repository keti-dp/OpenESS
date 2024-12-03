## 라이브러리
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pylab as plt 
import scipy.io
from scipy import stats
from scipy import io
from scipy.io import loadmat
from matplotlib.ticker import ScalarFormatter
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers



# 배터리 정상/과충전/과방전 모듈(6S2P) 전압, 전류 데이터 셋 기반 내부 모델 파라미터(Ri, Rdiff, Cdiff) 추출

# Load OCV data
OCV = '경로 설정 필요'
# 셀 level OCV는 OCV.csv 파일 사용

SOC = np.linspace(1, 0, num=21)

normal_dir = '경로 설정 필요'
overcharge_dir = '경로 설정 필요'
overdischarge_dir = '경로 설정 필요'
save_dir = '경로 설정 필요'

# 정상, 과충전 및 과방전 1~100사이클 충전 및 방전 구분해서 코드 실행 필요
# ex) 첨부된 정상, 과충전, 과방전 팩 100사이클 충전/방전으로 dir 설정해서 코드 실행
# 셀은 1~6열(셀 전압), 7열(모듈 전압), 8열(모듈 전류)



# 내부 모델 파라미터 추출할 셀 번호
cells_to_analyze = [1, 2, 3, 4, 5, 6]

# RLS 알고리즘 설계를 위한 초기 용량, 내부 모델 파라미터, 공분산 선정
Time = 1
Init_cap = 3.6
Init_Ri = 0.0310707902382320
Init_Rdiff = 0.0190371443335961
Init_Cdiff = 6093.35087066012
ErrorCovariance = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
ForgettingFactor = 0.99

Vector_b0 = Init_Ri
Vector_b1 = (-Init_Ri + (1 / Init_Cdiff) + (Init_Ri / (Init_Rdiff * Init_Cdiff)))
Vector_a1 = ((1 / (Init_Rdiff * Init_Cdiff)) - 1)

# 실제 모듈의 전압, 전류 데이터 기반 SOC-OCV 및 OCV-SOC Lookup table 설계
SOC_OCV_lookup = dict(zip(SOC, OCV))
OCV_SOC_lookup = dict(zip(OCV, SOC))

sorted_SOC_OCV_lookup = sorted(SOC_OCV_lookup.items())
sorted_OCV_SOC_lookup = sorted(OCV_SOC_lookup.items())

def find_soc_from_voltage(voltage, lookup_table):
    ocvs, socs = zip(*lookup_table)
    ocvs = np.array(ocvs)
    socs = np.array(socs)

    if voltage < ocvs.min():
        slope = (socs[1] - socs[0]) / (ocvs[1] - ocvs[0])
        extrapolated_soc = socs[0] + slope * (voltage - ocvs[0])
    elif voltage > ocvs.max():
        slope = (socs[-1] - socs[-2]) / (ocvs[-1] - ocvs[-2])
        extrapolated_soc = socs[-1] + slope * (voltage - ocvs[-1])
    else:
        extrapolated_soc = np.interp(voltage, ocvs, socs)
    
    return extrapolated_soc

def find_ocv_from_soc(soc, lookup_table):
    socs, ocvs = zip(*lookup_table)
    socs = np.array(socs)
    ocvs = np.array(ocvs)

    if soc < socs.min():
        slope = (ocvs[1] - ocvs[0]) / (socs[1] - socs[0])
        extrapolated_ocv = ocvs[0] + slope * (soc - socs[0])
    elif soc > socs.max():
        slope = (ocvs[-1] - ocvs[-2]) / (socs[-1] - socs[-2])
        extrapolated_ocv = ocvs[-1] + slope * (soc - socs[-1])
    else:
        extrapolated_ocv = np.interp(soc, socs, ocvs)
    
    return extrapolated_ocv

# 실시간 오차 공분산 및 gain 값 업데이트
# 배터리 시스템 level에 따라 covariance값 조정 필요
def calculate_gain_and_covariance_cell(phi, P, forgetting_factor):
    P_phi = P @ phi
    gain_denominator = forgetting_factor + phi.T @ P @ phi 
    gain = P_phi / gain_denominator
    covariance = (P - gain @ phi.T @ P) / forgetting_factor
    covariance = np.clip(covariance, 0, 0.001) 
    return gain, covariance
