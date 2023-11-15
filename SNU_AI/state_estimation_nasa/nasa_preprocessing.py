import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import utils.config as config


def preprocessing(df):
    # Modify current "I" (+- change)
    df['I'] = -df['I']
    
    # Define time difference "dt"
    df['dt'] = df['t'] - df['t'].shift(1)
    df.loc[0, 'dt'] = 0
    df.loc[df['dt'] > 50, 'dt'] = 0 # for some outliers

    # Define Q difference "dQ"
    df['dQ'] = df['I'] * df['dt']
    
    # Define "relative Q"
    df['relative Q'] = df['dQ'].cumsum()
    
    # Define "Q(mAh)"
    df['Q(mAh)'] = df['relative Q'] / 86400 * 1000
    
    # Define the real Q "Q"
    df['Q'] = 2100 + df['Q(mAh)'] # by the manual (README_RW_*.html)
    
    # Define cycle number "N"
    df['N'] = np.nan
    temp = df.loc[df['CDR'] != df['CDR'].shift(1)].copy()
    cycle_num = len(temp)
    temp.loc[:, 'N'] = [i+1 for i in range(cycle_num)]
    df.loc[temp.index, 'N'] = temp.loc[:, 'N']
    df = df.fillna(method='ffill')
    df['N'] = np.array(df['N'], dtype='int')
    
    # Define reference cycle number "ref_N"
    df['ref_N'] = np.nan
    temp = df.loc[(df['charge_type'] == 'reference charge') | (df['charge_type'] == 'reference discharge')]
    temp1 = temp.copy()
    temp = temp[temp['charge_type'] != temp['charge_type'].shift(1)]
    ref_num = len(temp)
    temp.loc[:, 'ref_N'] = [i+1 for i in range(ref_num)]
    temp1.loc[temp.index, 'ref_N'] = temp.loc[:, 'ref_N']
    temp1 = temp1.fillna(method='ffill')
    df.loc[:, 'ref_N'] = 0
    df.loc[temp1.index, 'ref_N'] = temp1.loc[:, 'ref_N']
    df['ref_N'] = np.array(df['ref_N'], dtype='int')
    
    # define SOC
    df['SOC'] = df['Q']/df['Q'].max()
    
    # define RW (random walk), cycle(from charge to next charge)
    df['RW'] = False
    temp = df.loc[df['charge_type'].str.contains('random')].copy()
    df.loc[temp.index, 'RW'] = True
    
    df['start_of_cycle'] = np.nan
    sample_rw = df[(df['RW'] == True) & (df['charge_type'] != df['charge_type'].shift(1))].copy()
    sample_ref = df[(df['RW'] == False) & (df['charge_type'] != df['charge_type'].shift(1))].copy()
    temp1 = sample_ref[(sample_ref['CDR'] == 'C')].copy()
    temp3 = sample_ref[(sample_ref['charge_type'].str.contains('pulse')) & ((sample_ref['charge_type'].shift(1).str.contains('reference discharge')))]
    temp2 = sample_rw[(sample_rw['CDR'] == 'C')].copy()
    df.loc[temp1.index, 'start_of_cycle'] = 1
    df.loc[temp2.index, 'start_of_cycle'] = 1
    df.loc[temp3.index, 'start_of_cycle'] = 1
         
    temp = df[df['start_of_cycle'] == 1].copy()
    df['cycle'] = np.nan
    df.loc[temp.index, 'cycle'] = [i+1 for i in range(len(temp))]
    df['cycle'].fillna(method = 'ffill', inplace = True)
    df['cycle'].fillna(0, inplace = True)
    df['cycle'] = np.array(df['cycle'], dtype = 'int')
    
    # define ref_cycle and ref_type
    df['ref_cycle'] = np.nan
    ref_index = df[(df['RW'] == False) & (df['cycle'] != df['cycle'].shift(1))].copy()

    df.loc[ref_index.index, 'ref_cycle'] = np.array(range(1, len(ref_index) + 1)) # reference_cycle's ref_cycle > 0
    df['ref_cycle'].fillna(method = 'ffill', inplace = True)
    df.loc[df['RW'] == True, 'ref_cycle'] = 0 # if cycle is not reference_cycle
    
    df['ref_type'] = np.nan   
    
    temp = df.loc[df['ref_cycle'] > 0]
    grouped = temp.groupby('cycle')
    for i, ind in enumerate(list(set(temp['cycle']))):
        sample = grouped.get_group(ind)
        for charge_type in list(set(sample['charge_type'].values)):
            if 'pulse' in charge_type:
                df.loc[sample.index, 'ref_type'] = 2 # if pulsed load is in this cycle
                break
            elif 'low' in charge_type:
                df.loc[sample.index, 'ref_type'] = 0 # low current discharge is in this cycle
                break
            else:
                df.loc[sample.index, 'ref_type'] = 1 # else, used to measure SOH, usually there are 2 cycles in one reference

    # define SOH by Q with scipy's interp1d
    df['SOH'] = np.nan
    temp = df[(df['start_of_cycle'] == 1) & (df['ref_type'] == 1)]
    
    SOH_list = []
    done_list = []
    for i in temp['cycle']:
        if i-1 in done_list:
            continue
        Q_list = df[(df['cycle'] == i) & (df['CDR'] == 'D')]['Q'] # discharge.max - discharge.min
        SOH_list.append(Q_list.max() - Q_list.min())
        done_list.append(i)
    f1 = interp1d(np.array(done_list), np.array(SOH_list), kind='linear', fill_value='extrapolate')
    start_point = df[df['start_of_cycle'] == 1]

    df.loc[start_point.index, 'SOH'] = f1(np.array(range(1, len(start_point) + 1)))
    
    # Capacitiy added.
    df['capacity'] = df['SOH']
    df['capacity'].fillna(method = 'ffill', inplace = True)
    
    df['SOH'] = df['SOH']/df['SOH'].max()
    df['SOH'].fillna(method = 'ffill', inplace = True)
    
    temp = df[(df['ref_cycle'] > 0) & (df['RW'].shift(1) != False)].copy()
    df['group'] = np.nan
    df.loc[temp.index, 'group'] = [i+1 for i in range(len(temp))]
    df['group'].fillna(method = 'ffill', inplace = True)
    df['group'] = np.array(df['group'], dtype = 'int')
    
    
    return df



if __name__ == "__main__":
    raw_data_dir = config.RAW_DATA_DIR
    processed_data_dir = config.PROCESSED_DATA_DIR
    
    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)
    
    for battery_id in config.BATTERY_NAME:
        try:
            battery_data = pd.read_parquet(os.path.join(raw_data_dir, battery_id))
            processed_data = preprocessing(battery_data)
            processed_data.to_parquet(os.path.join(processed_data_dir, battery_id))
        except Exception as e:
            print("Error occurs at", battery_id, "dataset.")