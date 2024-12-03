import pandas as pd
import os
import numpy as np

# 충전 구간에서의 HI추출
# source_folder 경로
source_folder = r"경로 설정"

# 이동 평균 적용 전류 데이터
def moving_average(data, window_size):
    return pd.Series(data).rolling(window=window_size).mean()

# 충전 시작 지점
def find_charge_start_point(current, voltage, check_window=3, ma_window=3):
    smoothed_current = moving_average(current, ma_window)
    for i in range(check_window, len(smoothed_current)):
        current_rising = smoothed_current[i] - smoothed_current[i - check_window] > 0
        voltage_rising = voltage[i] - voltage[i - check_window] > 0
        if current_rising and voltage_rising and smoothed_current[i] > 0:
            return i
    return None

# 충전 종료 지점
def find_charge_end_point(current, voltage, discharge_start):
    closest_point = None
    sustained_zero_start = None
    if discharge_start is not None:
        for i in range(discharge_start, 0, -1):
            if current[i-1] > 0 and current[i] <= 0:
                if closest_point is None:
                    closest_point = i
                zero_sustain_count = 0
                for j in range(i, len(current)):
                    if current[j] == 0:
                        zero_sustain_count += 1
                    else:
                        break
                if zero_sustain_count >= 20000 and np.all(np.diff(voltage[i:i + zero_sustain_count]) <= 0):
                    sustained_zero_start = i
                    break
    return sustained_zero_start if sustained_zero_start is not None else closest_point

# VIECTD 계산 함수
def calculate_viectd(charge_voltage, voltage_points, window_size):
    viectd_results = {}
    max_voltage = charge_voltage.max()
    for v_point in voltage_points:
        v_point_scaled = v_point * 240
        valid_indices = np.where(charge_voltage >= v_point_scaled)[0]
        if len(valid_indices) > 0:
            closest_index = valid_indices[0]
            closest_voltage = charge_voltage.iloc[closest_index]
            if closest_index + window_size < len(charge_voltage):
                future_voltage = charge_voltage.iloc[closest_index + window_size]
            else:
                future_voltage = charge_voltage.iloc[-1]
            viectd_results[v_point_scaled] = closest_voltage - future_voltage
        else:
            viectd_results[v_point_scaled] = np.nan
    return viectd_results

# TIECVD 계산 함수 (30s만 계산)
def calculate_tiecvd(charge_voltage, voltage_points, window_size=30):
    tiecvd_results = {}
    for i in range(1, len(voltage_points)):
        prev_v_point_scaled = voltage_points[i - 1] * 240
        curr_v_point_scaled = voltage_points[i] * 240
        prev_valid_indices = np.where(charge_voltage >= prev_v_point_scaled)[0]
        curr_valid_indices = np.where(charge_voltage >= curr_v_point_scaled)[0]
        if len(prev_valid_indices) > 0 and len(curr_valid_indices) > 0:
            prev_index = prev_valid_indices[0]
            curr_index = curr_valid_indices[0]
            tiecvd_results[curr_v_point_scaled] = np.abs(prev_index - curr_index)
        else:
            tiecvd_results[curr_v_point_scaled] = np.nan
    return tiecvd_results

# 데이터 처리 및 VIECTD, TIECVD 계산 함수
def process_ess_data(source_folder, file_names, voltage_points_viectd_30s, voltage_points_viectd_100s, voltage_points_tiecvd_30s):
    results = []
    for file_name in file_names:
        file_path = os.path.join(source_folder, file_name)
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            charge_voltage = df.iloc[:, 5]
            charge_current = df.iloc[:, 6]

            # 충전 구간 구분
            charge_start = find_charge_start_point(charge_current, charge_voltage)
            discharge_start = find_discharge_start(charge_current, charge_voltage)
            charge_end = find_charge_end_point(charge_current, charge_voltage, discharge_start)

            if charge_start is not None and charge_end is not None:
                charge_data = df.iloc[charge_start:charge_end]

                viectd_results_30s = calculate_viectd(charge_data.iloc[:, 5], voltage_points_viectd_30s, window_size=30)
                viectd_results_100s = calculate_viectd(charge_data.iloc[:, 5], voltage_points_viectd_100s, window_size=100)
                tiecvd_results_30s = calculate_tiecvd(charge_data.iloc[:, 5], voltage_points_tiecvd_30s, window_size=30)

                results.append({
                    "file_name": file_name,
                    "viectd_30s": viectd_results_30s,
                    "viectd_100s": viectd_results_100s,
                    "tiecvd_30s": tiecvd_results_30s
                })
    return results

# 설정: 전압 포인트 및 VIECTD, TIECVD 계산에 필요한 윈도우 크기
voltage_points_viectd_30s = np.arange(3.5, 4.0, 0.01)
voltage_points_viectd_100s = np.arange(3.5, 4.1, 0.1)
voltage_points_tiecvd_30s = np.arange(3.6, 4.2, 0.01)

# 20240101 ~ 20240331 기간의 모든 파일 처리 및 결과 저장
file_dates = pd.date_range(start="2024-01-01", end="2024-03-31")
file_names = [f'{date.strftime("%Y%m%d")}_rack.csv' for date in file_dates if os.path.exists(os.path.join(source_folder, f'{date.strftime("%Y%m%d")}_rack.csv'))]
results = process_ess_data(source_folder, file_names, voltage_points_viectd_30s, voltage_points_viectd_100s, voltage_points_tiecvd_30s)
