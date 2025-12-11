import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os,sys
import argparse

def soc_slope_synthesis(data, anomaly_intensity, output_dir, preprocessed = True):
    
    file_name = get_last_directory(data)
    data = pd.read_parquet(data)
    data = data.copy()


    if "BANK_SOC" not in data.columns:
        raise ValueError("The input data must contain a 'BANK_SOC' column.")
    
    
    start_idx, end_idx, expired_idx = search_discharge(data)
    
    updated_end_idx = np.ceil(end_idx - anomaly_intensity * 0.5 *(end_idx -start_idx)).astype(int)
    
    if preprocessed :
        soc_len = 1
    else :
        soc_len = 100
        
    start_val = data.iloc[start_idx]["BANK_SOC"]
    end_val = data.iloc[end_idx]["BANK_SOC"]
    # updated_val = data.iloc[updated_end_idx]["BANK_SOC"]
    
    slope_adjustment = soc_len * (end_val-start_val) * (1 / (updated_end_idx - start_idx) - 1 / (end_idx - start_idx))   
    for i in range(start_idx,end_idx) :
        """
        intensity 강도에 따른 비율을 기존의 SOC 방전구간값에서 제함. 
        배터리가 노화, 손상이 심할수록 방전속도가 빠를것이라 생각하여 합성의 intensity값이 클수록 방전속도가 빠르게 되도록 데이터 합성.
        
        """

        new_value = data["BANK_SOC"].iloc[i] + slope_adjustment * (i - start_idx)
        
        
        data.at[i, "BANK_SOC"] = max(data["BANK_SOC"][expired_idx+1], new_value)
        
        # if i%1000==0:
        #     breakpoint()
        
        
    
    if 'SOC_slope' not in data.columns:
        data['SOC_slope'] = calculate_soc_slope(data)
        
    print(f"intensity {anomaly_intensity} synthesize complete. SOC_slope value : {calculate_soc_slope(data)}")    
    
    if output_dir != None :
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        save_dir = output_dir +'/'+ os.path.splitext(file_name)[0] +'_SOC_slope_' + str(anomaly_intensity) + '.parquet'
        data.to_parquet(save_dir)
        print(f'output saved in {save_dir}.')
        
    return data

def get_last_directory(data_dir):
        return os.path.basename(os.path.normpath(data_dir))

def load_file(file_path):
    
    file_extension = os.path.splitext(file_path)[1].lower()  
    if file_extension == '.csv':
        print("CSV 파일로 인식했습니다.")
        return pd.read_csv(file_path)
    elif file_extension == '.parquet':
        print("Parquet 파일로 인식했습니다.")
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"지원되지 않는 파일 형식입니다: {file_extension}")
    
def search_discharge(data,threshold = 0.0001) :
    # 방전구간 탐색
    """
    방전구간 탐색 방법 변경 : data["BANK_SOC"].diff() < 0 을 탐색기준으로 할 경우 방전이 monotone decreasing이 아닐경우 문제가생겨 10타임스탬프의 평균을 기준으로 차이가 threshold 보다 크면 방전구간으로 정의.
    """
    discharge_indices = []
    
    for i in range(0,len(data),10):
        # if i>40000 and i%100==0:
        #     print(i)
        #     breakpoint()
        
        if np.sum(data["BANK_SOC"][i:i+10]) - np.sum(data["BANK_SOC"][i+10:i+20]) > threshold and i+20 in data.index :
            
            discharge_indices.append(i)
    
    filtered_discharge_indices = []
    for i, idx in enumerate(discharge_indices):
        # 슬라이딩 윈도우 범위 내에 있는 인덱스 세기
        nearby_count = 0
        for j in range(i+1, len(discharge_indices)):
            if abs(discharge_indices[j] - idx) <= 100:
                nearby_count += 1
            else:
                break
        
        # 3개 이상 근방 인덱스가 있을 경우
        if nearby_count >= 3:
            filtered_discharge_indices.append(idx)
            
    start_idx = filtered_discharge_indices[0]+5
    end_idx = filtered_discharge_indices[-1]+5
    expired_idx = discharge_indices[-1]
    
    
    return start_idx,end_idx,expired_idx

def calculate_soc_slope(data):
    
    if "BANK_SOC" not in data.columns:
        raise ValueError("The input data must contain a 'BANK_SOC' column.")

    start_idx, end_idx,_ = search_discharge(data)
    
    # discharge_length = (len(data)-start_idx-5000)
    
    scale = 8500
    
    # 구간을 10등분
    indices = np.linspace(start_idx, end_idx, 11, dtype=int)

    max_slope = 0

    # 구간별 slope 계산
    for i in range(len(indices) - 1):
        segment_start = indices[i]
        segment_end = indices[i + 1]

        # 구간 길이 0인경우 제외
        if segment_start == segment_end:
            continue

        # slope 계산
        delta_soc = data['BANK_SOC'][segment_end] - data['BANK_SOC'][segment_start]
        delta_time = segment_end - segment_start
        slope = abs(delta_soc / delta_time) * scale # or *discharge_length

        
        if slope > max_slope:
            max_slope = slope

    return max_slope
def main():
    # ArgumentParser 객체 생성
    parser = argparse.ArgumentParser(description="Process intensity and output directory.")

    # 인자 추가
    parser.add_argument('--intensity', type=float, required=True, help="Intensity value (from 0 to 1).")
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of input data")

    # 인자 파싱
    args = parser.parse_args() 
    
    soc_slope_synthesis(args.data_dir,args.intensity,args.output_dir)
    
if __name__ == "__main__":
    main()
    