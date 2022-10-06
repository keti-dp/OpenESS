import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def NASA_preprocess_basic(df):
    # Modify current I
    df['I'] = -df['I']
    
    # Time modification
    df['dt'] = df['t'] - df['t'].shift(1)
    df.loc[0, 'dt'] = 10
    weird = np.array(df.loc[df['dt'] > 1000].index)
    is_unknown_charge = df.loc[weird, 'V'].values > df.loc[weird-1, 'V'].values
    unknown_charge = weird[np.where(is_unknown_charge)[0]]
    unknown_discharge = weird[np.where(~is_unknown_charge)[0]]
    df.loc[:, 'unknown_mark'] = np.nan
    df.loc[unknown_charge-1, 'unknown_mark'] = 'unknown_charge_start'
    df.loc[unknown_charge, 'unknown_mark'] = 'unknown_charge_end'
    df.loc[unknown_discharge-1, 'unknown_mark'] = 'unknown_discharge_start'
    df.loc[unknown_discharge, 'unknown_mark'] = 'unknown_discharge_end'
    
    # Estimate capacity at each moment using reference discharge
    # These values are also used for estimating the charge-discharge rate
    # during the unknown interval.
    df.loc[:, 'capacity'] = np.nan
    temp = df.loc[df['charge_type'] == 'reference discharge']
    ref_D_start_idx = list(temp.loc[(temp['t'] - temp['t'].shift(1)) > 100, :].index)
    ref_D_end_idx = list(temp.loc[(temp['t'].shift(-1) - temp['t']) > 100, :].index)
    ref_D_start_idx.insert(0, temp.index[0])
    ref_D_end_idx.append(temp.index[-1])
    assert len(ref_D_start_idx) == len(ref_D_end_idx)

    for idx in range(0, len(ref_D_start_idx), 1): # Note: this does not take long time !
        start = ref_D_start_idx[idx]
        end = ref_D_end_idx[idx]
        discharge_I = temp.loc[start:end, 'I']
        discharge_dt = temp.loc[start:end, 'dt']
        exact_discharge_capa = -(discharge_I * discharge_dt).sum()
        df.loc[end, 'capacity'] = exact_discharge_capa
    
    temp = df.loc[ref_D_end_idx, :]
    assert len(temp.iloc[0::2, :]) == len(temp.iloc[1::2, :])
    capa_1 = temp.iloc[0::2, :].loc[:, 'capacity'].values
    capa_2 = temp.iloc[1::2, :].loc[:, 'capacity'].values
    capa_mean = (capa_1 + capa_2) / 2
    capa_1 = capa_mean * 1.0005
    capa_2 = capa_mean * 0.9995
    capa = np.concatenate([capa_1.reshape(-1, 1), capa_2.reshape(-1, 1)], axis=1).reshape(-1)
    df.loc[ref_D_end_idx, 'capacity'] = capa
    
    # Resampling timesteps -> interval : (exactly) 1 second
    # Naively, assume that the experiment starts 1st, Jan, 2021. (doesn't matter)
    df.index = df['t'] * 1e+9 + (1e+9 * 86400 * (365 * 51 + 13))
    df.index = pd.DatetimeIndex(df.index)
    df = df.resample('1S').first()
    
    df.loc[:, 'unknown_mark'] = df.loc[:, 'unknown_mark'].fillna(method='ffill')
    df.loc[:, 'unknown_mark'] = df.loc[:, 'unknown_mark'].replace('unknown_charge_end', np.nan)
    df.loc[:, 'unknown_mark'] = df.loc[:, 'unknown_mark'].replace('unknown_discharge_end', np.nan)
    df.loc[:, 'unknown_mark'] = df.loc[:, 'unknown_mark'].fillna(0)
    df.loc[:, 'charge_type'] = df.loc[:, 'charge_type'].fillna(0)
    idx = (df['charge_type'] == 0) & (df['unknown_mark'] != 0)
    df.loc[idx, 'charge_type'] = df.loc[idx, 'unknown_mark'].map(lambda x: ' '.join(x.split('_')[:2]))
    df.loc[idx & (df['charge_type'] == 'unknown charge'), 'CDR'] = 'C'
    df.loc[idx & (df['charge_type'] == 'unknown discharge'), 'CDR'] = 'D'
    
    df.loc[:, 'charge_type'] = df.loc[:, 'charge_type'].replace(0, np.nan)
    df.loc[:, 'charge_type'] = df.loc[:, 'charge_type'].fillna(method='ffill')
    
    df.loc[:, 'CDR'] = df.loc[:, 'CDR'].fillna(method='ffill')
    
    columns = ['V', 'I', 'charge_type', 'CDR', 'capacity']
    df = df.loc[:, columns]
    df.insert(0, 'index', np.arange(len(df)))
    
    # Interpolate 
    df.loc[:, 'capacity'] = df.loc[:, 'capacity'].fillna(0)
    temp = df.loc[df['capacity'] != 0, :]
    x = temp['index'].values
    y = temp['capacity'].values
    f = interp1d(x, y, fill_value='extrapolate')
    df.loc[:, 'capacity'] = f(df['index'].values)
    
    df.loc[:, 'V'] = df.loc[:, 'V'].interpolate()
    df.loc[:, 'I'] = df.loc[:, 'I'].interpolate()
    
    return df


def NASA_preprocess_advanced(df):

    # Define "SOH"
    df.loc[:, 'SOH'] = np.nan
    df.loc[:, 'SOH'] = df.loc[:, 'capacity'] / df.loc[:, 'capacity'].max()

    # Define cycle number "N"
    # Note : regard "C", "D+R" (consecutive) as one cycle
    df.loc[:, 'N'] = np.nan
    start_C = (df['CDR'] == 'C') & (df['CDR'].shift(1) != 'C')
    start_DR = (df['CDR'] != 'C') & (df['CDR'].shift(1) == 'C')
    cycle_start = start_C | start_DR
    df.loc[cycle_start, 'N'] = np.arange(1, cycle_start.sum()+1)
    df.loc[:, 'N'] = df.loc[:, 'N'].fillna(method='ffill')
    df.loc[:, 'N'] = df.loc[:, 'N'].fillna(0)
    df.loc[:, 'N'] = df.loc[:, 'N'].astype(np.int64)

    # Define RW (random walk)
    df.loc[:, 'RW'] = False
    df.loc[df['charge_type'].str.contains('random'), 'RW'] = True

    # Define UNK (unknown charge/discharge)
    df.loc[:, 'UNK'] = False
    df.loc[df['charge_type'].str.contains('unknown'), 'UNK'] = True

    # Define Ref_type (reference cycle type)
    # A :
    #   low current discharge at 0.04A
    #   rest prior/post low current discharge
    # B : 
    #   reference charge/discharge
    #   rest post reference charge/discharge
    # C :
    #   pulsed load (rest/discharge)
    #   rest post pulsed load
    df.loc[:, 'ref_type'] = np.nan
    df.loc[df['RW'], 'ref_type'] = 'none' # non-reference
    df.loc[df['UNK'], 'ref_type'] = 'none' # non-reference
    df.loc[df['charge_type'].str.contains('low'), 'ref_type'] = 'A'
    df.loc[df['charge_type'].str.contains('reference'), 'ref_type'] = 'B'
    df.loc[df['charge_type'].str.contains('pulsed'), 'ref_type'] = 'C'

    assert df['ref_type'].isna().sum() == 0

    # Define reference cycle number "ref_N"
    # Note : Only ref cycles of type B are considered for the ref-cycle number.
    # Note : if a cycle is non-reference or type A or type C, "ref_N" is -1. 
    df.loc[:, 'ref_N'] = np.nan
    df.loc[df['RW'], 'ref_N'] = -1 # non-reference
    df.loc[df['UNK'], 'ref_N'] = -1 # non-reference
    df.loc[df['ref_type'] == 'A', 'ref_N'] = -1
    df.loc[df['ref_type'] == 'C', 'ref_N'] = -1
    ref_start = (df['ref_type'] == 'B') & (df['ref_type'].shift(1) != 'B')
    df.loc[ref_start, 'ref_N'] = np.arange(1, ref_start.sum()+1)
    df.loc[:, 'ref_N'] = df.loc[:, 'ref_N'].fillna(method='ffill')
    
    # Define 'Q' (and also modify 'I' during 'unknwon charge')
    # Note : Integration of 'I' to obtain 'Q' is numerically unstable
    # Note : Reset SOC as 100% (i.e, 'Q' = 'capa') when one of the following:
    #   (1) 'charge (after random walk discharge)' ends
    #   (2) reference type A starts
    #   (3) reference type C starts
    df.loc[:, 'Q'] = np.nan
    first_capa = df.loc[:, 'capacity'].iloc[0]
    no_unk_df = df.loc[df['UNK'] == False, :].copy()
    no_unk_df.loc[:, 'Q'] = no_unk_df.loc[:, 'I'].cumsum() + first_capa

    # Reset SOC as 100% at particular points
    idx1 = (no_unk_df['RW'] == True) & (no_unk_df['CDR'] != 'C') & (no_unk_df['CDR'].shift(1) == 'C')
    idx2 = (no_unk_df['ref_type'] == 'A') & (no_unk_df['ref_type'].shift(1) != 'A')
    idx3 = (no_unk_df['ref_type'] == 'C') & (no_unk_df['ref_type'].shift(1) != 'C')
    idx = idx1 | idx2 | idx3
    err_Q = no_unk_df.loc[idx, 'capacity'] - no_unk_df.loc[idx, 'Q']
    temp = no_unk_df['Q'].copy()
    temp.loc[:] = np.nan
    temp.loc[idx] = err_Q temp = temp.fillna(method='ffill').fillna(0.)
    no_unk_df.loc[:, 'Q'] += temp # Modify the error

    # copy 'Q' frmo no_unk_df to df
    # (i.e, fill 'Q' in the 'unknown' interval using linear interpolation)
    df.loc[df['UNK'] == False, 'Q'] = no_unk_df.loc[:, 'Q']
    del no_unk_df
    df.loc[:, 'Q'] = df.loc[:, 'Q'].interpolate() # In the 'unknown' interval, linearly interpolate 'Q'

    # 'Q' min/max clipping
    df.loc[df['Q'] < 0, 'Q'] = 0.
    df.loc[df['Q'] > df['capacity'], 'Q'] = df.loc[:, 'capacity']

    # change 'I' on 'unknown charge/discharge'
    unk_idx = np.where(df['UNK'])[0]
    prev_unk_idx = unk_idx - 1
    diff = df.iloc[unk_idx, :].loc[:, 'Q'].values - df.iloc[prev_unk_idx, :].loc[:, 'Q'].values
    df.loc[df['UNK'], 'I'] = diff

    # Define 'SOC' by 'Q' / 'capacity'
    df.loc[:, 'SOC'] = df['Q'] / df['capacity']
    
    return df


if __name__ == '__main__':
    
    original_PATH = '/data2/ev/NASA_rand_parquet/'
    basic_PATH = '/data2/ev/NASA_rand_preprocessed_v2_basic/'
    advanced_PATH = '/data2/ev/NASA_rand_preprocessed_v2_advanced/'
    os.makedirs(basic_PATH, exist_ok=True)
    os.makedirs(advanced_PATH, exist_ok=True)
    
    for i in [3, 4, 5, 6]:
        data_file = os.path.join(original_PATH, f'RW{i:02d}.parquet')
        basic_file = os.path.join(basic_PATH, f'RW{i:02d}.parquet')
        advanced_file = os.path.join(advanced_PATH, f'RW{i:02d}.parquet')
        
        df = pd.read_parquet(data_file)
        df = NASA_preprocess_basic(df)
        df.to_parquet(basic_file, engine='pyarrow')
        df = NASA_preprocess_advanced(df)
        df.to_parquet(advanced_file, engine='pyarrow')
