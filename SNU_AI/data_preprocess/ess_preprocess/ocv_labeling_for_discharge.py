import datetime, os, argparse
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from functools import reduce

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



def OCV_label(data_path, site, voltage_threshold=None, save_data_path=None, save_fig_dir=None, monitor=False):
    '''
    data_path : data path for OCV labeling
    site : sionyu or panli
    voltage_thres : voltage threshold for sunny days (=fully-charged day)
    save_data_path : path for ocv labeled data (parquet or csv)
    save_fig_dir : save figure 1)clear days 2)daily OCV labeled 
        if None, do not save figure
    monitor : print moitoring conditions - discharge voltage drops
    '''
    if '.csv' == data_path[-4:]:
        file_name = data_path[:-4]
    elif '.parquet' == data_path[-8:]:
        file_name = data_path[:-8]

    data = pd.read_parquet(data_path)
    data = data.sort_values('TIMESTAMP')
    data = data.reset_index(drop=True)

    print(f'{file_name} OCV labeling starts')

    if site == 'sionyu':
        voltage_thres = 808 if voltage_threshold == None else voltage_threshold
        bank_capacity, b2c_current, b2c_voltage, ocv2soc, soc2ocv = sionyu_spec()
        data_ocv_labeled = ocv_labeling(data, site, voltage_thres, bank_capacity, b2c_current, b2c_voltage, ocv2soc, soc2ocv, save_fig_dir, file_name, monitor=monitor)

    if site == 'panli':
        voltage_thres = 970 if voltage_threshold == None else voltage_threshold
        
        data_ocv = []
        for bank_id_panli in [1, 2]:
            print(f"BANK ID : {bank_id_panli}")
            data_i = data[data['BANK_ID']== bank_id_panli]
            bank_capacity, b2c_current, b2c_voltage, ocv2soc, soc2ocv = panli_spec(bank_id=bank_id_panli)
            ocv_labeled = ocv_labeling(data_i, site, voltage_thres, bank_capacity, b2c_current, b2c_voltage, ocv2soc, soc2ocv, save_fig_dir, file_name, monitor=monitor)
            data_ocv.append(ocv_labeled)
        data_ocv_labeled = pd.concat([data_ocv[0], data_ocv[1]])

    if save_data_path == None:
        save_data_path = file_name + '_OCVlabeled.parquet'
        
    if '.csv' == save_data_path[-4:]:
        data_ocv_labeled.to_csv(save_data_path)
    elif '.parquet' == save_data_path[-8:]:
        data_ocv_labeled.to_parquet(save_data_path)
        
#         os.makedirs(save_data_dir+f'/{file_name}', exist_ok=True)
#         data_ocv_labeled.to_parquet(save_data_dir+f'/{file_name}/{file_name}.parquet')

    return data_ocv_labeled



def ocv_labeling(data, site, voltage_thres, bank_capacity, b2c_current, b2c_voltage, ocv2soc, soc2ocv, save_fig_dir, file_name, monitor, scaling=True):
    # Prepare data. fully-charged day
    thres_above = data[['TIMESTAMP', 'BANK_DC_VOLT']][data['BANK_DC_VOLT'] >= voltage_thres]
    clear_date = pd.to_datetime(thres_above['TIMESTAMP'], utc=True).dt.tz_convert('Asia/Seoul').dt.date.unique()

    clear_data = []
    for date in clear_date:
        clear_data.append(select_oneday_data(data, date=date))

        
    # plot clear_data
    if save_fig_dir != None:
        os.makedirs(save_fig_dir+f'/{file_name}', exist_ok=True)
        plt.figure(figsize=(16,12), facecolor='white')
        for oneday in clear_data:
            plt.plot(oneday['BANK_DC_VOLT'].to_numpy(), alpha=0.2, label=oneday['TIMESTAMP'].iloc[0])
        plt.legend()
        plt.title(file_name)
        plt.savefig(save_fig_dir + f'/{file_name}/{file_name}_{len(clear_date)}cleardays')

    
    # extract discharge data
    bank_labeled_clear = pd.DataFrame({})
    for oneday in clear_data:
        oneday_date = str(oneday['TIMESTAMP'].iloc[0])[:10]

        discharge_data = extract_discharge(oneday, site)
        discharge_data['AVERAGE_CELL_VOLT'] = discharge_data['BANK_DC_VOLT'] / b2c_voltage
        
        if monitor:
            print(discharge_data['AVERAGE_CELL_VOLT'].iloc[0], 'to', discharge_data['AVERAGE_CELL_VOLT'].iloc[-1]) ## monitor

        capacity = calculated_capacity(discharge_data)

        soc_init = ocv2soc(discharge_data['AVERAGE_CELL_VOLT'].iloc[0])
        
        # Scale capacity.
        if scaling:
            soc_end = ocv2soc(oneday['BANK_DC_VOLT'].iloc[-1] / b2c_voltage)
            scale_factor = (soc_end - soc_init) / (capacity[-1] / bank_capacity)
        else:
            scale_factor = 1
            
        soc_calculated = 100 * (soc_init + capacity / bank_capacity * scale_factor)

        ocv_estimated = soc2ocv(soc_calculated / 100) * b2c_voltage
        discharge_data['OCV_est'] = ocv_estimated
        

        if save_fig_dir != None:
            os.makedirs(save_fig_dir+f'/{file_name}/daily', exist_ok=True)
            plt.figure(figsize=(16,12), facecolor='white')
            ax1 = plt.subplot()
            ax1.plot(discharge_data['TIMESTAMP'], ocv_estimated, label='ocv_estimated')
            ax1.plot(oneday['TIMESTAMP'], oneday['BANK_DC_VOLT'], label='bank_voltage')
            plt.legend(loc=2)

            ax2 = ax1.twinx()
            ax2.plot(oneday['TIMESTAMP'], oneday['BANK_SOC'], color='green', label='SOC label')
            plt.legend(loc=1)

            plt.title(oneday_date)
            plt.savefig(save_fig_dir+f'/{file_name}/daily/' + oneday_date)
            # plt.show()

        # oneday = oneday[['TIMESTAMP', 'OCV_est']]
        oneday_labeled = pd.merge(oneday, discharge_data, how='inner', on='TIMESTAMP')
        oneday_labeled = oneday_labeled[['TIMESTAMP', 'OCV_est']]

        bank_labeled_clear = pd.concat([bank_labeled_clear, oneday_labeled], axis=0)

    bank_labeled = pd.merge(data, bank_labeled_clear, how='inner', on='TIMESTAMP')

    return bank_labeled



def extract_discharge(oneday, site):
    if site == 'sionyu':
        charge_end = oneday[oneday['BATTERY_STATUS_FOR_CHARGE'] == 2].index[-1]
        _oneday = oneday.loc[charge_end:, ['BATTERY_STATUS_FOR_CHARGE']]

        first_index = _oneday[_oneday['BATTERY_STATUS_FOR_CHARGE'] == 3].index[0]
        last_index = _oneday[_oneday['BATTERY_STATUS_FOR_CHARGE'] == 3].index[-1]

    elif site == 'panli':
        first_index = oneday[oneday['BATTERY_STATUS_FOR_DISCHARGE'] == 1].index[0] #- 1
        last_index = oneday[oneday['BATTERY_STATUS_FOR_DISCHARGE'] == 1].index[-1] #+ 1
        oneday_discharge = oneday.loc[first_index:last_index]
    
        if sum(oneday_discharge['BATTERY_STATUS_FOR_CHARGE'] == 1) > 0:
            first_index = oneday_discharge[oneday_discharge['BATTERY_STATUS_FOR_STANDBY'] == 1].index[-1]
            oneday_discharge = oneday.loc[first_index:last_index]
    
    discharge_data = oneday.loc[first_index:last_index, :]

    return discharge_data



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
    parser.add_argument("--data_path", required=True, type=str, help='Path for data to label OCV values; ends with .parquet or .csv')
    parser.add_argument("--site", default='sionyu', type=str, help='Name of the BMS site; sionyu/panli')
    parser.add_argument("--voltage_threshold", default=None, type=float, help='Voltage threshold for sunny days')
    parser.add_argument("--save_path", default=None, type=str, help='Path for saving data; ends with .parquet or .csv')
    
    options = parser.parse_args()
    OCV_label(options.data_path, options.site, options.voltage_threshold, options.save_path)