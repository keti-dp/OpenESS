import os, argparse, datetime
import pandas as pd
import matplotlib.pyplot as plt
from utils.ocv_labeling_total import OCV_label

RAW_COLUMNS_SIONYU = [
    'TIMESTAMP',
    'BANK_DC_VOLT',
    'BANK_DC_CURRENT',
    'BANK_SOC',
    'MAX_CELL_TEMPERATURE_OF_BANK',
    'VOLT_gap',
    'OCV_est',
    'BATTERY_STATUS_FOR_CHARGE'
]

COL2NAME_SIONYU = {
    'TIMESTAMP' : 'time',
    'BANK_DC_VOLT' : 'V',
    'BANK_DC_CURRENT' : 'I',
    'BANK_SOC' : 'SOC',
    'MAX_CELL_TEMPERATURE_OF_BANK' : 'T',
    'VOLT_gap' : 'V_gap',
    'OCV_est' : 'OCV',
    'BATTERY_STATUS_FOR_CHARGE' : 'status'
}

NAME2COL_SIONYU = {
    'time' : 'TIMESTAMP',
    'V' : 'BANK_DC_VOLT',
    'I' : 'BANK_DC_CURRENT',
    'SOC' : 'BANK_SOC',
    'T' : 'MAX_CELL_TEMPERATURE_OF_BANK',
    'V_gap' : 'VOLT_gap',
    'OCV' : 'OCV_est',
    'status' : 'BATTERY_STATUS_FOR_CHARGE'
}

def select_oneday_data(data, date=None, year=None, month=None, day=None, local_timezone=True):
    """
    data : data for all day
    date : date when data is collected, type=datetime.date or tuple of (year, month, day)
    year, month, day : date when data is collected, used when date is None
    local_timezone : True for local timezone(Asia/Seoul), False for UTC
    """
    if date == None:
        date = datetime.date(year, month, day)
    elif not isinstance(date, datetime.date):
        date = datetime.date(*date)
    timezone = 'Asia/Seoul' if local_timezone else 'UTC'
    return data[pd.to_datetime(data['TIMESTAMP'], utc=True).dt.tz_convert(timezone).dt.date == date]

def choose_day(data, dates):
    """
    data : data for all day
    dates : 남기려고 하는 날짜 list of int
    """
    total_dates = pd.to_datetime(data['TIMESTAMP'], utc=True).dt.tz_convert('Asia/Seoul').dt.date.unique()
    result = []
    if not isinstance(dates, list):
        dates = list(dates)
    dates = [int(date) for date in dates]
    
    for date in total_dates:
        if date.day in dates:
            result.append(select_oneday_data(data, date=date))
    
    return pd.concat(result, axis=0)

def get_dates():
    # V plot 기준 데이터가 없거나 제대로 충전하고 있다고 판단되지 않는 날들을 제외
    dates = {}
    dates['sionyu'] = {}
    dates['sionyu']['bank'] = {}
    dates['panli'] = {}
    dates['panli']['bank1'] = {}
    dates['panli']['bank2'] = {}

    dates['sionyu']['bank'][2] = [2, 3, 8, 9, 10, 11, 12, 13, 14, 17, 18, 21, 22, 23, 24, 25, 26] # 17 days
    dates['sionyu']['bank'][3] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 21, 22, 23, 24, 27, 28, 29, 30, 31] # 21 days
    dates['sionyu']['bank'][4] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 27, 28, 30] # 24 days
    dates['sionyu']['bank'][5] = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31] # 26 days
    dates['sionyu']['bank'][6] = [1, 2, 3, 4, 5, 6, 25, 30] # 8 days
    dates['sionyu']['bank'][7] = [1, 2, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 25, 26, 27, 28, 29, 30] # 23 days
    # total 119 days

    dates['panli']['bank1'][3] = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] # 24 days
    dates['panli']['bank1'][4] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30] # 30 days
    dates['panli']['bank1'][5] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] # 30 days
    dates['panli']['bank1'][6] = [1, 2, 3, 4, 5, 26, 27, 28, 29] # 9 days
    dates['panli']['bank1'][7] = [5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30] # 22 days
    # total 115 days

    dates['panli']['bank2'][3] = [8, 9, 10, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] # 16 days
    dates['panli']['bank2'][4] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30] # 30 days
    dates['panli']['bank2'][5] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] # 30 days
    dates['panli']['bank2'][6] = [1, 2, 3, 4, 5, 26, 27, 28, 29] # 9 days
    dates['panli']['bank2'][7] = [5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30] # 22 days
    # total 107 days

    return dates

def preprocess(site, filename, month):
    base_dir = '/data/ess/data/incell/state_estimation_data'
    data_dir = os.path.join(base_dir, site, 'preprocessed', filename)
    data = pd.read_parquet(data_dir)
    dates = get_dates()
    usable_columns = ['TIMESTAMP', 'BANK_DC_VOLT', 'BANK_DC_CURRENT', 'BANK_SOC', 'MAX_CELL_TEMPERATURE_OF_BANK', 'VOLT_gap', 'OCV_est', 'BATTERY_STATUS_FOR_CHARGE']
    
    if site == 'panli':
        if 'bank1' in filename:
            bank = 'bank1'
        elif 'bank2' in filename:
            bank = 'bank2'
        else:
            raise ValueError('filename error')
    else:
        bank = 'bank'

    filename = filename.split('.')[0]
    print(f'{site} {filename} preprocessing starts')

    deleted_data = choose_day(data, dates[site][bank][month])
    ocv_labeled_data = OCV_label(deleted_data, site, bank)
    ocv_labeled_data['VOLT_gap'] = ocv_labeled_data['MAX_CELL_VOLTAGE_OF_BANK'] - ocv_labeled_data['MIN_CELL_VOLTAGE_OF_BANK']

    base_save_data_path = '/data/ess/data/incell/state_estimation_data'
    save_data_path = os.path.join(base_save_data_path, site, 'ocv_labeled', filename + '_OCVlabeled.parquet')
        
    ocv_labeled_data[usable_columns].to_parquet(save_data_path)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--site", default='sionyu', type=str, help='Name of the BMS site; sionyu/panli')
    # parser.add_argument("--filename", required=True, type=str, help='Path for data; ends with .parquet or .csv')
    # parser.add_argument("--month", required=True, type=int, help='Month of data')
    
    # args = parser.parse_args()
    # preprocess(args.site, args.filename, args.month)

    
    sionyu_base_path = '/data/ess/data/incell/state_estimation_data/sionyu/preprocessed'
    panli_base_path = '/data/ess/data/incell/state_estimation_data/panli/preprocessed'
    sionyu_data_path = os.listdir(sionyu_base_path)
    sionyu_data_path.sort()
    panli_data_path = os.listdir(panli_base_path)
    panli_data_path.sort()
    
    for data_path, month in zip(sionyu_data_path, [2, 3, 4, 5, 6, 7]):
        preprocess('sionyu', data_path, month)

    for data_path, month in zip(panli_data_path, [3, 3, 4, 4, 5, 5, 6, 6, 7, 7]):
        preprocess('panli', data_path, month)
