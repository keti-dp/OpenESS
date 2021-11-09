import numpy as np
import pandas as pd
from config import min_soc, max_soc, soc_margin, input_size

def get_capacity(batt_data, discharging = False):
    capacity = 0
    for i,t in zip(batt_data['Current_measured'],batt_data['Time']):
        capacity += i*t
    
    if discharging:
        return -capacity
    else:
        return capacity

def get_Rf(volt_curr_data, init_data = True):
    eod = np.argmin(volt_curr_data['Voltage_measured'])
    volt_eod = volt_curr_data['Voltage_measured'][eod]
    curr_eod = volt_curr_data['Current_measured'][eod]
    sor = np.where(volt_curr_data['Current_measured'] > -0.1)[0]
    for s in sor:
        if s > eod + 5:
            volt_sor = volt_curr_data['Voltage_measured'][s]
            return (volt_eod - volt_sor) / curr_eod
        

def get_SOC(batt_data,capacity):
    soc = [0]
    length = len(batt_data['Current_measured'])
    for k, i, t in zip(range(1,length+1), batt_data['Current_measured'], batt_data['Time']):
        now_soc = soc[k-1] + (i*t / capacity)
        soc.append(now_soc)
    return soc

def get_Vsei(data, r):
    data['Voltage_measured'] = np.array(data['Voltage_measured'])
    # nan 값 수정
    nan_idx = np.argwhere(pd.isnull(data['Voltage_measured']))
    for nidx in nan_idx:
        data['Voltage_measured'][nidx] = (data['Voltage_measured'][nidx-1] + data['Voltage_measured'][nidx+1]) / 2
        data['Current_measured'][nidx] = (data['Current_measured'][nidx-1] + data['Current_measured'][nidx+1]) / 2
        data['Temperature_measured'][nidx] = (data['Temperature_measured'][nidx-1] + data['Temperature_measured'][nidx+1]) / 2

    cap = get_capacity(data)
    soc = get_SOC(data,cap)

    volt_threshold = np.mean(data['Voltage_measured']) * 0 # 미정 // Vsei를 시작할 부분
    init_threshold = np.where(data['Voltage_measured'] > volt_threshold)[0]
    vsei = []
    temperature = []
    pre_soc = 0
    
    for i in init_threshold:
        if soc[i] > min_soc and soc[i] < max_soc and soc[i] >= (pre_soc + soc_margin):
            v = data['Voltage_measured'][i]
            curr = data['Current_measured'][i]
            vsei.append(v - r * curr)
            temperature.append(data['Temperature_measured'][i])
            pre_soc = soc[i]

        if len(vsei) > input_size-1:
            feature_set = []
            for j in range(len(vsei)-1):
                feature_set.append(vsei[j+1] -vsei[j])
            feature_set.append(np.mean(temperature))
            return feature_set
    print("error")
    exit()

def get_SOH(cmax,c): # 안됨
    return c/cmax * 100