import os, time, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional, Union


# Site별 module 개수
num_modules = {
    "1": 17, 
    "2": 20,
    "3a": 17,
    "4a": 20
}

# min/max scaling 범위 지정 (1 cell 당 기준으로)
ylims = {
    "BANK_DC_VOLT": [3.2, 4.2],
    "BANK_DC_CURRENT": [-3.5, 3.5],
    "BANK_SOC": [0, 100],
    "MAX_CELL_TEMPERATURE_OF_BANK": [5, 45],
    "VOLTAGE_GAP": [0, 0.4],
}

def has_column_names(df):
    # Check if the columns are a RangeIndex or empty
    return not (df.columns.equals(pd.RangeIndex(start=0, stop=len(df.columns))) or df.columns.isnull().any())

def preprocess_timestamp_and_columns(root_dir, dataset_name, save_root_dir):
    """
    작성자 : 최명제
    preprocessing for incell ESS data
    Args:
        root_dir (str) : 전처리할 df 파일이 모여있는 곳. ex) root_dir/year/month/date/yyyymmdd_bank.parquet에서 root_dir
        dataset_name (str) : 시온유 - 1, 판리 - 2, 황금6호 - 3a, 백마 - 4a
        save_root_dir (str) : 전처리한 df 파일을 모을 곳. ex) root_dir/dataset_name/yyyymmdd.parquet에서 root_dir
    """
    t0 = time.time()
    error_list = []
    
    # 저장경로 생성
    if save_root_dir[-1]!="/": 
        new_save_dir = save_root_dir+"/"
    else:
        new_save_dir = save_root_dir
    save_dir = new_save_dir+dataset_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 각 연월일별로 순회를 돌며 df 구축
    years = sorted(os.listdir(root_dir))
    for year in years:
        year_path = os.path.join(root_dir, year)
        months = sorted(os.listdir(year_path))
        for month in months:
            print(year+month)
            path = os.path.join(year_path, month)
            dates = sorted([name for name in os.listdir(path)])
            date_dirs = sorted([os.path.join(path, name) for name in os.listdir(path)])
            
            # df 파일 열기 (parquet -> ft, feather -> csv 순서)
            for date, name in zip(dates, date_dirs):
                assert name[-2:] == date
                parquet_save_path = f"{save_dir}/{year}{month}{date}.parquet"
                
                # 이미 전처리 되어있는 경우 지나감 (새로 데이터 들어온 경우 대비)
                if os.path.exists(parquet_save_path):
                    continue
                
                if os.path.exists(os.path.join(name, year+month+date+"_bank.parquet")):
                    df_path = os.path.join(name, year+month+date+"_bank.parquet")
                    df = pd.read_parquet(df_path, engine="pyarrow")
                elif os.path.exists(os.path.join(name, year+month+date+"_bank.ft")):
                    df_path = os.path.join(name, year+month+date+"_bank.ft")
                    df = pd.read_feather(df_path)
                elif os.path.exists(os.path.join(name, year+month+date+"_bank.feather")):
                    df_path = os.path.join(name, year+month+date+"_bank.feather")
                    df = pd.read_feather(df_path)
                elif os.path.exists(os.path.join(name, year+month+date+"_bank.csv")):
                    df_path = os.path.join(name, year+month+date+"_bank.csv")
                    try:
                        df = pd.read_csv(df_path)
                    except:
                        error_list.append("empty_csv,"+year+month+date) # csv 파일이 껍데기만 있는 경우
                        continue
                else:
                    error_list.append("no_df_file,"+year+month+date) # 폴더에 df 관련 파일이 없는 경우
                    continue
                
                # column이 없는 df 예외처리
                if not has_column_names(df):
                    error_list.append("no_columns,"+year+month+date)
                    continue
                
                # column에 Voltage 등 정보가 없는 경우 예외처리
                try:
                    df_ = df["BANK_DC_VOLT"].copy()
                    df_ = 0
                except:
                    error_list.append("no_columns,"+year+month+date)
                    continue
                
                # BANK=1인 것만 처리 (추후 수정 가능)
                if len(np.unique(df["BANK_ID"])) != 1:
                    df = df[df["BANK_ID"]==1].copy().reset_index(drop=True)
                
                # 길이가 짧은 df 제거 (계측 길이 부족)
                if len(df) < 83000:
                    error_list.append("short_df,"+year+month+date)
                    continue
                
                ###########################################
                # 자료값 전처리
                mins = []           # 사전에 정한 min값보다 낮으면 리스트에 넣음. len(mins)>0 이면 예외처리
                maxes = []          # 사전에 정한 max값보다 높으면 리스트에 넣음. len(maxes)>0 이면 예외처리
                small_diffs = []    # 두 값의 차이가 거의 없으면 리스트에 넣음.   len(small_diffs)>0 이면 예외처리
                                    # 적용 대상 : BANK_DC_VOLTAGE, BANK_DC_CURRENT, BANK_SOC, threshold = 0.1
                not_sunny = []      # 흐린 날, 비 온 날로 추정되면 리스트에 넣음. 기준: max(BANK_SOC) < 0.5
                # Voltage Gap 구현
                df["VOLTAGE_GAP"] = df["MAX_CELL_VOLTAGE_OF_BANK"] - df["MIN_CELL_VOLTAGE_OF_BANK"]
                
                columns = ["TIMESTAMP", "BATTERY_STATUS_FOR_CHARGE", "BANK_DC_VOLT", "BANK_DC_CURRENT", "BANK_SOC", "MAX_CELL_TEMPERATURE_OF_BANK", "VOLTAGE_GAP"]
                df = df[columns]
                
                # 초단위로 가공 및 중간 빈 위치 interpolate
                # 적용 마무리 단계에서 df.reset_index 해야 함 (마지막 단계에 있음)
                # missing_timestamps : interpolate할 위치. 현재는 미사용, 혹시나 해서 남겨둠
                df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP']).dt.floor("S")
                df = df.drop_duplicates(subset=['TIMESTAMP'])
                df = df.set_index('TIMESTAMP')
                df = df.asfreq('S')
                # missing_timestamps = df[df["BANK_DC_VOLT"].isna()].index
                
                # TIMESTAMP, BATTERY_STATUS_FOR_CHARGE가 아닌 column들에 대하여 scaling 실시
                for col in columns[2:]: 
                    df[col] = df[col].interpolate()
                    if col in ["BANK_DC_VOLT", "BANK_DC_CURRENT"]:
                        df[col] = df[col]/num_modules[dataset_name]/12
                    df[col] = (df[col] - ylims[col][0]) / (ylims[col][1] - ylims[col][0])
                    min_, max_ = np.min(df[col]), np.max(df[col])
                    if min_<0:
                        mins.append(col)
                    if max_>1:
                        maxes.append(col)
                    if col in ["BANK_DC_VOLT", "BANK_DC_CURRENT", "BANK_SOC"] and max_ - min_ < 0.1:
                        small_diffs.append(col)
                    if col == "BANK_SOC" and max_ < 0.5:
                        not_sunny.append(col)
                
                # 전처리 기준을 만족하지 못한 df는 걸러냄.
                if len(mins) > 0:
                    error_list.append(f"out_of_min_range_{mins[0]},"+year+month+date)
                    continue
                if len(maxes) > 0:
                    error_list.append(f"out_of_max_range_{maxes[0]},"+year+month+date)
                    continue
                if len(small_diffs) > 0:
                    error_list.append(f"too_small_differences_{small_diffs[0]},"+year+month+date)
                    continue
                if len(not_sunny) > 0:
                    error_list.append("not_sunny,"+year+month+date)
                    continue
            
                df = df.reset_index()
                if len(df) == 86401: # len(df) = 86401인 경우 다음날 0시0분 내용이 포함된 것
                    df = df.iloc[:-1, :]
                
                # Voltage Gap 이상치 합성을 위한 내용
                df["VOLTAGE_GAP_CURVE"] = get_voltage_gap_curve(df["BANK_DC_CURRENT"])

                
                # 전처리 완료된 df를 parquet 형태로 저장
                df.to_parquet(parquet_save_path)
                
    if len(error_list) == 0:
        print("no errors")
    else:
        # 예외처리 항목 저장
        error_list = np.unique(error_list).tolist()
        error_list = [n+"\n" for n in error_list]
        with open(os.path.join(save_root_dir, f"errors_{dataset_name}.txt"), "w") as f:
            f.writelines(error_list)
    
    print(f"Elapsed Time : {time.time() - t0}")
                    
def plot_preprocessed_parquets(root_path: str, 
                               save_plot_path: Optional[str] = None,
                               save_names_path: Optional[str] = None):
    """
    작성자 : 최명제
    plotting columns of preprocessed dataframe for each site.
    Args:
        root_path (str) : preprocessed dataframe이 모여있는 경로. ex) root_path/dataset_name/yyyymmdd.parquet
        save_plot_path (Optional[str]) : if save_plot_path != None, 해당 위치에 plot 저장
        save_names_path (Optional[str]) : if save_plot_path != None, 해당 위치에 names(디렉토리+parquet이름), short_names(parquet이름)을 pickle로 저장
    """
    folders = os.listdir(root_path)
    folders = sorted([name for name in folders if not name.endswith(".txt")])

    names = []
    short_names = []

    for folder in folders:
        print(folder) 
        # minmax를 모을 dict 생성
        minmax = {
            "BANK_DC_VOLT": [[], []],
            "BANK_DC_CURRENT": [[], []],
            "BANK_SOC": [[], []],
            "MAX_CELL_TEMPERATURE_OF_BANK": [[], []],
            "VOLTAGE_GAP": [[], []],
            "df_length": []
            }
        
        path = os.path.join(root_path, folder)
        parquet_list = os.listdir(path)
        parquet_list = [name for name in parquet_list if name.endswith("parquet")]
        parquet_list = sorted(parquet_list)
        short_names.append(parquet_list)
        parquet_list = [os.path.join(path, name) for name in parquet_list]
        names.append(parquet_list)
        
        # 각 df에 대해 순회를 돌며 각 column의 min, max를 수집
        print("reading parquets...")
        for parquet_path in tqdm(parquet_list):
            df = pd.read_parquet(parquet_path, engine="pyarrow")
            
            minmax["df_length"].append(len(df))
            for col in ["BANK_DC_VOLT", "BANK_DC_CURRENT", "BANK_SOC", "MAX_CELL_TEMPERATURE_OF_BANK", "VOLTAGE_GAP"]:
                min_, max_ = np.min(df[col]), np.max(df[col])
                minmax[col][0].append(min_)
                minmax[col][1].append(max_)
                
        print("start plotting...")
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        for i, col in enumerate(["BANK_DC_VOLT", "BANK_DC_CURRENT", "BANK_SOC", "MAX_CELL_TEMPERATURE_OF_BANK", "VOLTAGE_GAP"]):
            min_, max_ = np.min(minmax[col][0]), np.max(minmax[col][1])
            X = list(range(len(minmax[col][0])))
            axs[i//3, i%3].scatter(X, minmax[col][0], s=1)
            axs[i//3, i%3].scatter(X, minmax[col][1], s=1)
            axs[i//3, i%3].set_title(col)
            axs[i//3, i%3].set_ylim([0, 1])
        
        axs[-1, -1].plot(minmax["df_length"])
        axs[-1, -1].set_title("df_length")
            
        fig.suptitle(f"min/max statistics of {folder}", fontsize=25, fontweight="bold")
        if save_plot_path is not None:
            fig.savefig(f"statistics_{folder}_before", bbox_inches="tight")
        
        with open(f"statistics_{folder}.pickle", "wb") as f:
            pickle.dump(minmax, f)
    
    if save_names_path is not None:
        with open(f"short_names.pickle", "wb") as f:
            pickle.dump(short_names, f)
        with open(f"names.pickle", "wb") as f:
            pickle.dump(names, f)

# Voltage Gap
def find_steep_change_indices(input_array, threshold):
    change_indices = []
    for i in range(1, len(input_array)):
        # Compare the current element with the previous one
        if input_array[i] - input_array[i - 1]>threshold:
            # If the value changes, record the index
            change_indices.append(i)
    return change_indices

def get_voltage_gap_curve(I,
                          rest_voltage_difference = -0.06, 
                          peak_max_voltage = 0.16, 
                          peak_min_voltage = -0.1):
    newVgap = np.zeros_like(I)
    VgapMax = ylims["VOLTAGE_GAP"][1]
    R_btw_DC_voltage = np.random.normal(rest_voltage_difference, 0.01)/VgapMax
    peak_max_voltage = np.random.normal(peak_max_voltage, 0.01)/VgapMax
    peak_min_voltage = np.random.normal(peak_min_voltage, 0.01)/VgapMax
    criterion = np.array(I[(I<0.45) | (I>0.55)].index)
    change_idx = find_steep_change_indices(criterion, 10)[0]
    changes = [criterion[0], criterion[change_idx-1], criterion[change_idx], criterion[-1]]
    newVgap[:changes[0]]=R_btw_DC_voltage
    newVgap[changes[0]:changes[1]] = np.linspace(R_btw_DC_voltage, peak_max_voltage, changes[1]-changes[0])
    newVgap[changes[1]:changes[2]] = peak_max_voltage
    newVgap[changes[2]:changes[-1]] = np.linspace(peak_max_voltage, peak_min_voltage, changes[-1]-changes[2])
    midpoint = int(np.mean([changes[-1], 86400]))
    l = midpoint-changes[-1]
    a = peak_min_voltage - R_btw_DC_voltage
    a = a / l / l
    xx = a*(np.array(range(l))-l)**2 + R_btw_DC_voltage
    newVgap[changes[-1]:midpoint] = xx
    newVgap[midpoint:] = R_btw_DC_voltage
    return newVgap
    
def synthesize_voltage_gap(V, I, Vgap, 
                           rest_voltage_difference = -0.06, 
                           peak_max_voltage = 0.16, 
                           peak_min_voltage = -0.1):
    """
    작성자 : 최명제
    synthesize anomalous voltage gap via observed pattern.
    Args:
        V (pd.Series) : Series of voltage (BANK_DC_VOLT)
        I (pd.Series) : Series of current (BANK_DC_CURRENT)
        Vgap (pd.Series) : Series of voltage (VOLTAGE_GAP)
        rest_voltage_difference (float) : cell voltage difference of broken cell before charge (while resting)
        peak_max_voltage (float) : max cell voltage difference of broken cell while charging
        peak_min_voltage (float) : min cell voltage difference of broken cell while discharging
    Returns:
         VgapFinal (pd.Series) : Synthesized anomalous voltage gap.
    """
    newVgap = get_voltage_gap_curve(I, rest_voltage_difference, peak_max_voltage, peak_min_voltage)
    newV = V + newVgap
    VgapFinal = abs(newV-V)+Vgap
    
    return VgapFinal

# DEPRECATED
# 혹시나 해서 남겨둠

# def preprocess(root_dir, dataset_name, save_root_dir):
#     """
#     processing function for 2, 3a
#     """
#     t0 = time.time()
#     error_list = []
#     years = os.listdir(root_dir)
#     for year in years:
#         year_path = os.path.join(root_dir, year)
#         months = os.listdir(year_path)
#         for month in months:
#             print(year+month)
#             path = os.path.join(year_path, month)
#             dates = sorted([name for name in os.listdir(path)])
#             date_dirs = sorted([os.path.join(path, name) for name in os.listdir(path)])
            
#             for count, col in enumerate(columns[:-1]):
#                 fig, axs = plt.subplots(5, 7, figsize=(21, 10))
                
#                 for i, (date, name) in enumerate(zip(dates, date_dirs)):
#                     assert name[-2:] == date
#                     # load df
#                     if os.path.exists(os.path.join(name, year+month+date+"_bank.parquet")):
#                         df_path = os.path.join(name, year+month+date+"_bank.parquet")
#                         df = pd.read_parquet(df_path, engine="pyarrow")
#                     elif os.path.exists(os.path.join(name, year+month+date+"_bank.ft")):
#                         df_path = os.path.join(name, year+month+date+"_bank.ft")
#                         df = pd.read_feather(df_path)
#                     elif os.path.exists(os.path.join(name, year+month+date+"_bank.feather")):
#                         df_path = os.path.join(name, year+month+date+"_bank.feather")
#                         df = pd.read_feather(df_path)
#                     elif os.path.exists(os.path.join(name, year+month+date+"_bank.csv")):
#                         df_path = os.path.join(name, year+month+date+"_bank.csv")
#                         try:
#                             df = pd.read_csv(df_path)
#                         except:
#                             error_list.append("empty_csv,"+year+month+date)
#                             continue
#                     else:
#                         error_list.append("no_df_file,"+year+month+date)
#                         continue
                    
#                     if not has_column_names(df):
#                         error_list.append("no_columns,"+year+month+date)
#                         continue
                    
#                     try:
#                         df_ = df["BANK_DC_VOLT"].copy()
#                         df_ = 0
#                     except:
#                         error_list.append("no_columns,"+year+month+date)
#                         continue
                    
#                     if len(np.unique(df["BANK_ID"])) != 1:
#                         df = df[df["BANK_ID"]==1].copy().reset_index(drop=True)
                        
#                     if len(df) < 83000:
#                         error_list.append("short_df,"+year+month+date)
#                         continue

#                     if type(col) == tuple:
#                         df["VOLTAGE_GAP"] = (df[col[1]] - df[col[0]]).copy()
#                         axs[i//7, i%7].plot(df["VOLTAGE_GAP"])
#                         axs[i//7, i%7].set_ylim(ylims["VOLTAGE_GAP"])
#                     elif count == 0 or count == 1:
#                         df[col] = df[col]/num_modules[dataset_name]/12
#                         axs[i//7, i%7].plot(df[col])
#                         axs[i//7, i%7].set_ylim(ylims[col])
#                     else:
#                         axs[i//7, i%7].plot(df[col])
#                         axs[i//7, i%7].set_ylim(ylims[col])
#                     axs[i//7, i%7].set_title(f"{month}/{date}")
#                     axs[i//7, i%7].set_xticks([])
                        
#                 if type(col) == tuple:
#                     fig.suptitle(f"{dataset_name}-{year}-{month}-VOLTAGE_GAP", fontsize=25, fontweight="bold")
#                     fig.savefig(os.path.join(save_root_dir, f"{dataset_name}-{year}-{month}-VOLTAGE_GAP"), bbox_inches="tight")
#                 else:
#                     fig.suptitle(f"{dataset_name}-{year}-{month}-{col}", fontsize=25, fontweight="bold")
#                     fig.savefig(os.path.join(save_root_dir, f"{dataset_name}-{year}-{month}-{col}"),bbox_inches="tight")
#                 plt.close()
#     if len(error_list) == 0:
#         print("no errors")
#     else:
#         error_list = np.unique(error_list).tolist()
#         error_list = [n+"\n" for n in error_list]
#         with open(os.path.join(save_root_dir, "errors.txt"), "w") as f:
#             f.writelines(error_list)
#     print(f"Elapsed Time : {time.time() - t0}")
    

