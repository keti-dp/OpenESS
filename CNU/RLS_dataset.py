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

def process_condition_for_cycle(condition_dir, cycle_num, condition_name):
    cycle_filename = f'Cycle_{cycle_num}.csv'
    condition_data = pd.read_csv(os.path.join(condition_dir, cycle_filename))
    
    # Initialize lists to store results for this cycle
    Ri_estimates_all_cells = []
    Rdiff_estimates_all_cells = []
    Cdiff_estimates_all_cells = []
    OCV_all_cells = []
    SOC_all_cells = []  # SOC 저장할 리스트 추가
    
    for cell in cells_to_analyze:
        Volin = condition_data.iloc[:, [cell - 1]]  # Use cell 1 to 6 voltage data
        Curin = condition_data.iloc[:, [7]] / 2  # Current is halved as it's divided by 2 in the original code

        Init_soc = [find_soc_from_voltage(vol, sorted_OCV_SOC_lookup) for vol in Volin.iloc[:, 0]]

        SOC_ref = []
        current_SOC = Init_soc[0]

        for i in range(len(Curin)):
            delta_SOC = (Curin.iloc[i, 0] / Init_cap) * (Time / 3600)
            current_SOC += delta_SOC
            SOC_ref.append(current_SOC)

        OCV_ref = [find_ocv_from_soc(soc, sorted_SOC_OCV_lookup) for soc in SOC_ref]

        theta = np.array([Vector_b0, Vector_b1, Vector_a1]).reshape(-1, 1)
        P = ErrorCovariance

        Ri_estimates = [Init_Ri]
        Rdiff_estimates = [Init_Rdiff]
        Cdiff_estimates = [Init_Cdiff]

        overpotential_values = []

        for t in range(1, len(Volin)):
            voltage = Volin.iloc[t, 0]
            current = Curin.iloc[t, 0]
            prev_current = Curin.iloc[t-1, 0]
            ocv = OCV_ref[t]

            overpotential = voltage - ocv
            overpotential_values.append(overpotential)

            overpotential_prev = overpotential_values[-2] if t > 1 else 0
            
            phi = np.array([current, prev_current, overpotential_prev]).reshape(-1, 1)
            
            gain, covariance = calculate_gain_and_covariance_cell(phi, P, ForgettingFactor)
            P = covariance

            RlsEstimatedVoltage = theta.T @ phi

            RlsVoltageError = voltage - ocv - RlsEstimatedVoltage

            theta = theta + gain @ RlsVoltageError
            theta = np.abs(theta)

            Ri = np.abs(theta[0, 0])
            Rdiff = (np.abs(theta[1, 0]) -  (np.abs(theta[2, 0]) * np.abs(theta[0, 0]))) / (1 - np.abs(theta[2, 0]))
            Cdiff = 1 / (np.abs(theta[1, 0]) - (np.abs(theta[2, 0]) * np.abs(theta[0, 0])))

            Ri = np.abs(Ri)
            Rdiff = np.abs(Rdiff)
            Cdiff = np.abs(Cdiff)
            
            Rdiff_estimates.append(Rdiff)
            Cdiff_estimates.append(Cdiff)
            Ri_estimates.append(Ri)

        Ri_estimates_all_cells.append(Ri_estimates)
        Rdiff_estimates_all_cells.append(Rdiff_estimates)
        Cdiff_estimates_all_cells.append(Cdiff_estimates)
        OCV_all_cells.append(OCV_ref)
        SOC_all_cells.append(SOC_ref)  # SOC 값을 각 셀별로 저장
    
    return Ri_estimates_all_cells, Rdiff_estimates_all_cells, Cdiff_estimates_all_cells, OCV_all_cells, SOC_all_cells

from datetime import datetime, timedelta

# Process and save data for each cycle with TIMESTAMP
def save_results_for_cycle_with_timestamp(cycle_num, Ri_all, Rdiff_all, Cdiff_all, OCV_all, SOC_all, condition_name):
    save_path = os.path.join(save_dir, f'{condition_name}_Cycle_{cycle_num}_Results.csv')
    
    # 2023년 10월 1일을 시작 날짜로 설정
    start_date = datetime(2023, 10, 1)
    
    # 각 사이클마다 하루씩 날짜를 추가
    cycle_date = start_date + timedelta(days=cycle_num - 1)
    
    with open(save_path, 'w') as f:
        f.write('TIMESTAMP,Cell,Ri,Rdiff,Cdiff,OCV,SOC\n')  # 헤더에 Timestamp 추가
        for cell_index in range(6):
            for t in range(len(Ri_all[cell_index])):
                # t초마다 시간을 기록하도록 변경
                TIMESTAMP = cycle_date + timedelta(seconds=t)  # t초마다 1초씩 증가된 시간
                
                # Timestamp 포맷은 YYYY-MM-DD HH:MM:SS
                TIMESTAMP_str = TIMESTAMP.strftime('%Y-%m-%d %H:%M:%S')
                
                # CSV 파일에 TIMESTAMP와 데이터를 기록
                f.write(f'{TIMESTAMP_str},{cell_index + 1},{Ri_all[cell_index][t]},{Rdiff_all[cell_index][t]},{Cdiff_all[cell_index][t]},{OCV_all[cell_index][t]},{SOC_all[cell_index][t]}\n')

# Process and save data for each cycle with TIMESTAMP
for cycle_num in range(1, 101):
    # Process normal condition
    Ri_normal, Rdiff_normal, Cdiff_normal, OCV_normal, SOC_normal = process_condition_for_cycle(normal_dir, cycle_num, 'normal')
    save_results_for_cycle_with_timestamp(cycle_num, Ri_normal, Rdiff_normal, Cdiff_normal, OCV_normal, SOC_normal, 'normal')
    
    # Process overcharge condition
    Ri_overcharge, Rdiff_overcharge, Cdiff_overcharge, OCV_overcharge, SOC_overcharge = process_condition_for_cycle(overcharge_dir, cycle_num, 'overcharge')
    save_results_for_cycle_with_timestamp(cycle_num, Ri_overcharge, Rdiff_overcharge, Cdiff_overcharge, OCV_overcharge, SOC_overcharge, 'overcharge')
    
    # Process overdischarge condition
    Ri_overdischarge, Rdiff_overdischarge, Cdiff_overdischarge, OCV_overdischarge, SOC_overdischarge = process_condition_for_cycle(overdischarge_dir, cycle_num, 'overdischarge')
    save_results_for_cycle_with_timestamp(cycle_num, Ri_overdischarge, Rdiff_overdischarge, Cdiff_overdischarge, OCV_overdischarge, SOC_overdischarge, 'overdischarge')
