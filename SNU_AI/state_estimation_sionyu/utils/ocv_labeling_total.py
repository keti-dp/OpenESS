import datetime, os, argparse
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from functools import reduce
from tqdm import tqdm

pd.set_option('mode.chained_assignment',  None)

# Select the data in the given date.
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

def OCV_label_save(data_path, site, save_data_path, monitor=False):
    '''
    data_path : data path for OCV labeling
    site : sionyu or panli
    monitor : print moitoring conditions - discharge voltage drops
    '''
    if '.csv' == data_path[-4:]:
        file_name = data_path.split('/')[-1][:-4]
    elif '.parquet' == data_path[-8:]:
        file_name = data_path.split('/')[-1][:-8]

    data = pd.read_parquet(data_path)
    # file_name = file_name.split('.')[0]
    print(f'{file_name} OCV labeling starts')
    data_ocv_labeled = OCV_label(data, site, data_path[-13:-8])

    # data = data.sort_values('TIMESTAMP')
    # data = data.reset_index(drop=True)


    # if site == 'sionyu':
    #     bank_capacity, _, b2c_voltage, ocv2soc, soc2ocv = sionyu_spec()
    #     data_ocv_labeled = ocv_labeling(data, site, bank_capacity, b2c_voltage, ocv2soc, soc2ocv, monitor=monitor)

    # if site == 'panli':
    #     if data_path[-13:-8] == 'bank1':
    #         bank_capacity, _, b2c_voltage, ocv2soc, soc2ocv = panli_spec(bank_id=1)
    #         data_ocv_labeled = ocv_labeling(data, site, bank_capacity, b2c_voltage, ocv2soc, soc2ocv, monitor=monitor)
    #     elif data_path[-13:-8] == 'bank2':
    #         bank_capacity, _, b2c_voltage, ocv2soc, soc2ocv = panli_spec(bank_id=2)
    #         data_ocv_labeled = ocv_labeling(data, site, bank_capacity, b2c_voltage, ocv2soc, soc2ocv, monitor=monitor)
    #     else:
    #         data_ocv = []
    #         for bank_id_panli in [1, 2]:
    #             print(f"BANK ID : {bank_id_panli}")
    #             data_i = data[data['BANK_ID'] == bank_id_panli]
    #             bank_capacity, _, b2c_voltage, ocv2soc, soc2ocv = panli_spec(bank_id=bank_id_panli)
    #             ocv_labeled = ocv_labeling(data_i, site, bank_capacity, b2c_voltage, ocv2soc, soc2ocv, monitor=monitor)
    #             data_ocv.append(ocv_labeled)
    #         data_ocv_labeled = pd.concat([data_ocv[0], data_ocv[1]])

    
    if save_data_path == None:
        base_save_data_path = '/data/ess/data/incell/state_estimation_data'
        save_data_path = os.path.join(base_save_data_path, site, 'ocv_labeled', file_name + '_OCVlabeled.parquet')
        
    if '.csv' == save_data_path[-4:]:
        data_ocv_labeled.to_csv(save_data_path)
    elif '.parquet' == save_data_path[-8:]:
        data_ocv_labeled.to_parquet(save_data_path)


def OCV_label(data, site, bank=None, monitor=False):
    '''
    data : data for OCV labeling
    site : sionyu or panli
    bank : bank of panli; bank1 or bank2
    monitor : print moitoring conditions - discharge voltage drops
    '''

    data = data.sort_values('TIMESTAMP')
    data = data.reset_index(drop=True)

    if site == 'sionyu':
        print('Get Sionyu battery spec.')
        bank_capacity, _, b2c_voltage, ocv2soc, soc2ocv = sionyu_spec()
        soc = np.arange(0., 1., 0.05)
        ocv = soc2ocv(soc) * b2c_voltage
        print('SOC is translated to OCV as following table.')
        print()
        print('SOC -> OCV')
        for _soc, _ocv in zip(soc, ocv):
            print(f' {int(_soc*100):2d} -> {_ocv:.3f}')
        print()
        print('Calculate OCV.')
        data_ocv_labeled = ocv_labeling(data, site, bank_capacity, b2c_voltage, ocv2soc, soc2ocv, monitor=monitor)

    if site == 'panli':
        if bank == 'bank1':
            bank_capacity, _, b2c_voltage, ocv2soc, soc2ocv = panli_spec(bank_id=1)
            data_ocv_labeled = ocv_labeling(data, site, bank_capacity, b2c_voltage, ocv2soc, soc2ocv, monitor=monitor)
        elif bank == 'bank2':
            bank_capacity, _, b2c_voltage, ocv2soc, soc2ocv = panli_spec(bank_id=2)
            data_ocv_labeled = ocv_labeling(data, site, bank_capacity, b2c_voltage, ocv2soc, soc2ocv, monitor=monitor)
        else:
            raise ValueError('bank value error')

    return data_ocv_labeled


def ocv_labeling(data, site, bank_capacity, b2c_voltage, ocv2soc, soc2ocv, monitor):
    # Prepare data. fully-charged day
    # thres_above = data[['TIMESTAMP', 'BANK_DC_VOLT']][data['BANK_DC_VOLT'] >= voltage_thres]
    # clear_date = pd.to_datetime(thres_above['TIMESTAMP'], utc=True).dt.tz_convert('Asia/Seoul').dt.date.unique()
    clear_date = pd.to_datetime(data['TIMESTAMP'], utc=True).dt.tz_convert('Asia/Seoul').dt.date.unique()
    
    clear_data = []
    for date in clear_date:
        clear_data.append(select_oneday_data(data, date=date))
    
    # extract discharge data
    bank_labeled_clear = pd.DataFrame({})
    for oneday in tqdm(clear_data):
        rest_data_1, charge_data, rest_data_2, discharge_data, rest_data_3 = split_data(oneday)
        try:
            charge_data = ocv_estimate(charge_data, discharge_data['BANK_DC_VOLT'].iloc[0], bank_capacity, b2c_voltage, ocv2soc, soc2ocv, monitor, charge=True)
        except IndexError:
            breakpoint()
        discharge_data = ocv_estimate(discharge_data, oneday['BANK_DC_VOLT'].iloc[-1], bank_capacity, b2c_voltage, ocv2soc, soc2ocv, monitor, charge=False)
        rest_data_1['OCV_est'] = charge_data['OCV_est'].iloc[0]
        rest_data_2['OCV_est'] = discharge_data['OCV_est'].iloc[0]
        rest_data_3['OCV_est'] = discharge_data['OCV_est'].iloc[-1]
        
        oneday_labeled = pd.concat([rest_data_1, charge_data, rest_data_2, discharge_data, rest_data_3], axis=0)
        oneday_labeled = oneday_labeled[['TIMESTAMP', 'OCV_est']]

        bank_labeled_clear = pd.concat([bank_labeled_clear, oneday_labeled], axis=0)
        
    bank_labeled = pd.merge(data, bank_labeled_clear, how='inner', on='TIMESTAMP')

    return bank_labeled


def split_data(oneday):
    charge_start, charge_end = oneday[oneday['BATTERY_STATUS_FOR_CHARGE'] == 2].index[[0,-1]]
    discharge_start, discharge_end = oneday[oneday['BATTERY_STATUS_FOR_CHARGE'] == 3].index[[0,-1]]

    return oneday.loc[:charge_start-1,:], oneday.loc[charge_start:charge_end,:], oneday.loc[charge_end+1:discharge_start-1,:], oneday.loc[discharge_start:discharge_end,:], oneday.loc[discharge_end+1:,:]


def sionyu_spec(inter_k='cubic'):
    '''
    battery spec : 33J
    1 bank = 8 rack (parallel), 1 rack = 17 module (serial), 1 module = 12 core (serial), 1 core = 60 cell (parallel)

    inter_k : interpolatge method
    '''
    ess_comp = {'bank': [8,'parallel'], 'rack': [17, 'serial'], 'module': [12, 'serial'], 'core': [60, 'parallel']}

    b2c_current =  reduce(lambda x, y : x*y, dict(filter(lambda v : v[1]=='parallel', ess_comp.values())).keys()) # 60*8
    b2c_voltage = reduce(lambda x, y : x*y, dict(filter(lambda v : v[1]=='serial', ess_comp.values())).keys()) #17*12

    bank_capacity = 2.962 * 60  * 8 # 33J spec * b2c_current
    
    # 33J spec
    soc_table = np.arange(1., 0, -0.05)
    ocv_table = np.array([4.036, 3.985, 3.941, 3.899, 3.842,
                          3.809, 3.778, 3.731, 3.693, 3.672,
                          3.656, 3.637, 3.626, 3.615, 3.599,
                          3.577, 3.548, 3.504, 3.463, 3.427])
    ocv2soc_main = interpolate.interp1d(ocv_table, soc_table, kind=inter_k)
    ocv2soc_end = interpolate.interp1d(np.array([3.427, 3.213]), np.array([0.05, 0]), kind='linear')
    soc2ocv_main = interpolate.interp1d(soc_table, ocv_table, kind=inter_k)
    soc2ocv_end = interpolate.interp1d(np.array([0.05, 0]), np.array([3.427, 3.213]), kind='linear')
    
    ocv2soc = lambda x : ocv2soc_main(x) if (x > 3.427) else ocv2soc_end(x)
    soc2ocv = lambda x : np.concatenate((soc2ocv_main(x[x > 0.05]), soc2ocv_end(x[x <= 0.05])))

    return bank_capacity, b2c_current, b2c_voltage, ocv2soc, soc2ocv



def panli_spec(bank_id, inter_k='cubic'):
    '''
    battery spec : 41J
    [BANK_ID 1] 1 bank = 9 rack (parallel), 1 rack = 10 module (serial), 1 module = 12 core (serial), 1 core = 60 cell (parallel)
    [BANK_ID 2] 1 bank = 8 rack (parallel), 1 rack = 10 module (serial), 1 module = 12 core (serial), 1 core = 60 cell (parallel)

    bank_id : 1 or 2
    '''
    if bank_id == 1:
        ess_comp = {'bank': [9,'parallel'], 'rack': [20, 'serial'], 'module': [12, 'serial'], 'core': [60, 'parallel']}
    else:
        ess_comp = {'bank': [8,'parallel'], 'rack': [20, 'serial'], 'module': [12, 'serial'], 'core': [60, 'parallel']}
    
    b2c_current =  reduce(lambda x, y : x*y, dict(filter(lambda v : v[1]=='parallel', ess_comp.values())).keys()) 
    b2c_voltage = reduce(lambda x, y : x*y, dict(filter(lambda v : v[1]=='serial', ess_comp.values())).keys()) 

    bank_capacity = 3.54 * b2c_current
    
    # 41J spec
    soc_table = np.arange(1., 0, -0.05)
    ocv_table = np.array([4.058, 4.002, 3.952, 3.909, 3.870,
                          3.829, 3.792, 3.731, 3.678, 3.650,
                          3.631, 3.615, 3.601, 3.586, 3.567,
                          3.539, 3.503, 3.450, 3.418, 3.378])
    ocv2soc_main = interpolate.interp1d(ocv_table, soc_table, kind=inter_k)
    ocv2soc_end = interpolate.interp1d(np.array([3.378, 2.851]), np.array([0.05, 0]), kind='linear')
    soc2ocv_main = interpolate.interp1d(soc_table, ocv_table, kind=inter_k)
    soc2ocv_end = interpolate.interp1d(np.array([0.05, 0]), np.array([3.378, 2.851]), kind='linear')
    
    ocv2soc = lambda x : ocv2soc_main(x) if (x > 3.378) else ocv2soc_end(x)
    soc2ocv = lambda x : np.concatenate((soc2ocv_main(x[x > 0.05]), soc2ocv_end(x[x <= 0.05])))

    return bank_capacity, b2c_current, b2c_voltage, ocv2soc, soc2ocv


def ocv_estimate(data, final_voltage, bank_capacity, b2c_voltage, ocv2soc, soc2ocv, monitor, charge, scaling=True):
    data['AVERAGE_CELL_VOLT'] = data['BANK_DC_VOLT'] / b2c_voltage
        
    if monitor:
        print(data['AVERAGE_CELL_VOLT'].iloc[0], 'to', data['AVERAGE_CELL_VOLT'].iloc[-1]) ## monitor

    capacity = calculated_capacity(data)

    soc_init = ocv2soc(data['AVERAGE_CELL_VOLT'].iloc[0])
    
    # Scale capacity.
    if scaling:
        soc_end = ocv2soc(final_voltage / b2c_voltage)
        scale_factor = (soc_end - soc_init) / (capacity[-1] / bank_capacity)
    else:
        scale_factor = 1
        
    soc_calculated = 100 * (soc_init + capacity / bank_capacity * scale_factor)

    if charge:
        ocv_estimated = b2c_voltage * soc2ocv(soc_calculated[::-1] / 100)[::-1]
    else:
        ocv_estimated = soc2ocv(soc_calculated / 100) * b2c_voltage
    data['OCV_est'] = ocv_estimated

    return data[['TIMESTAMP', 'OCV_est']]


def calculated_capacity(oneday):
    # Simpson rule
    oneday_even = oneday['BANK_DC_CURRENT'].iloc[::2].to_numpy()
    oneday_odd = oneday['BANK_DC_CURRENT'].iloc[1::2].to_numpy()
    base_time = pd.to_datetime(oneday['TIMESTAMP'].iloc[:1]).values.astype('datetime64[s]')[0]
    times = (pd.to_datetime(oneday['TIMESTAMP']).values - base_time).astype('float64') / 1000000000
    times_delta = times[1:] - times[:-1]

    odd_subint = True if len(oneday_even) == len(oneday_odd) else False
    _oneday_odd = oneday_odd[:-1] if odd_subint else oneday_odd
    _times_delta = times_delta[:-1] if odd_subint else times_delta

    _times_delta_block = _times_delta[::2] + _times_delta[1::2]
    cumsum_even = (1/6) * _times_delta_block * ((2 - _times_delta[1::2] / _times_delta[::2]) * oneday_even[:-1]\
                                                + _times_delta_block * _times_delta_block * _oneday_odd / (_times_delta[::2] * _times_delta[1::2])\
                                                + (2 - _times_delta[::2] / _times_delta[1::2]) * oneday_even[1:])

    cumsum_even = np.cumsum(cumsum_even) / 3600
    
    h_1 = times_delta[2::2]
    h_2 = times_delta[1:-1:2]

    h_plus = h_1 + h_2
    h_mul = h_1 * h_2
    h_1_sq = h_1 * h_1

    alpha = (2 * h_1_sq + 3* h_mul) / (6 * h_plus)
    beta = (h_1_sq + 3* h_mul) / (6 * h_2)
    gamma = h_1_sq * h_1 / (6 * h_2 * (h_plus))

    half_n = len(oneday_odd)
    cumsum_odd = cumsum_even[:half_n-1] + (alpha * oneday_odd[1:] + beta * oneday_even[1:half_n] - gamma * oneday_odd[:-1]) / 3600

    capacity = np.zeros_like(times)
    capacity[1] = (times[1] - times[0]) * (oneday_even[0] + oneday_odd[1]) / 2 / 3600
    capacity[2::2] = cumsum_even
    capacity[3::2] = cumsum_odd
    
    return capacity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # /data/ess/data/incell/state_estimation_data/sionyu/preprocessed/
    parser.add_argument("--data_path", required=True, type=str, help='Path for data to label OCV values; ends with .parquet or .csv')
    parser.add_argument("--site", default='sionyu', type=str, help='Name of the BMS site; sionyu/panli')
    parser.add_argument("--save_path", default=None, type=str, help='Path for saving data; ends with .parquet or .csv')
    
    options = parser.parse_args()
    OCV_label_save(options.data_path, options.site, options.save_path)
