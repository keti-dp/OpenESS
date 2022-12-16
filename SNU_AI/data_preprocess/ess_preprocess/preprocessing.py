"""
Script for data preprocessing

Last Updated : 22-01-09 by 정윤기
"""

import datetime, os
import numpy as np
import pandas as pd
import pandas.io.sql as psql
# import psycopg2 as pg
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate


DATA_DIR = '/data2/ess/data/'  # base data directory



# Fetch and save data from the server.
def fetch_data(table_name, file_name=None, first_date=None, last_date=None):
    """
    table_name : one of ['bank', 'rack', 'pcs', 'etc']
    file_name : name of file to save data, default name is 'DATA_DIR/<table_name>/<today>_<table_name>.csv'
    first_date : the first date in fetched data, type=datetime.date or tuple of (year, month, day)
    last_date : the day after last date in fetched data, type=datetime.date or tuple of (year, month, day)
    """
    assert table_name in ['bank', 'rack', 'pcs', 'etc']
    name = file_name if file_name != None else DATA_DIR + table_name +'/' + datetime.date.today().strftime('%y%m%d')\
                                                          + '_' + table_name + '.parquet'
    
    if file_name == None:
        if not os.path.exists('data/'):
            os.mkdir('data/')
        if not os.path.exists('data/' + table_name):
            os.mkdir('data/' + table_name)
    
    with pg.connect("postgres://guest_user:guest1234!@1.214.41.250:5434/ESS_Operating_Site1") as conn:
        sql = "SELECT * FROM " + table_name + ";"
        df = psql.read_sql(sql, conn)
    
    if first_date != None:
        if not isinstance(first_date, datetime.date):
            first_date = datetime.date(*first_date)
        df = df[pd.to_datetime(df['TIMESTAMP'], utc=True).dt.tz_convert('Asia/Seoul').dt.date >= first_date]
            
    if last_date != None:
        if not isinstance(last_date, datetime.date):
            last_date = datetime.date(*last_date)
        df = df[pd.to_datetime(df['TIMESTAMP'], utc=True).dt.tz_convert('Asia/Seoul').dt.date < last_date]
    
    df.to_parquet(name)
    return df
        
        
        
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



# Count the number of data in seconds 
def get_count_in_seconds(data, first_time=None, last_time=None):
    """
    data : data to count number of data itself in seconds
    first_time : the first time when counting starts, default is the first time in data, type : datetime64[s]
    last_time : the last time when counting ends, default is the last time in data, type : datetime64[s]
    """
    data_count = data['TIMESTAMP'].groupby(pd.to_datetime(data['TIMESTAMP']).values.astype('datetime64[s]')).count()
    time = data_count.index[0] if first_time == None else first_time
    end_time = data_count.index[-1] if last_time == None else last_time - datetime.timedelta(seconds=1)
    time_series = [time]
    
    while time < end_time:
        time = time + datetime.timedelta(seconds=1)
        time_series.append(time)
        
    time_count = pd.Series(index=time_series, data=0)
    time_count = (time_count + data_count).fillna(0).astype(int)
        
    return time_count



# Log invalid time data. (no data or multiple data in a second)
def log_invalid_time(time_count, file_name=None):
    """
    time_count : return from get_count_in_second
    file_name : name of log file, just print logs if file_name == None
    """
    no_data_time = time_count[time_count == 0].index
    multiple_data_time = time_count[time_count > 1].index
    
    no_data_logs = ['<No Data>\n']
    if len(no_data_time) > 0:
        start_time = no_data_time[0]
        current_time = start_time
        for time in no_data_time[1:]:
            if time - current_time == datetime.timedelta(seconds=1):
                current_time = time
            else:
                no_data_logs.append(str(start_time) + ' ~ ' + str(current_time)\
                                    + ', for ' + str((current_time - start_time).seconds + 1) + 's\n')
                start_time = time
                current_time = time
        no_data_logs.append(str(start_time) + ' ~ ' + str(current_time)\
                            + ', for ' + str((current_time - start_time).seconds + 1) + 's\n')
    
    multiple_data_logs = ['<Multiple Data>\n']
    if len(multiple_data_time) > 0:
        start_time = multiple_data_time[0]
        current_time = start_time
        for time in multiple_data_time[1:]:
            if time - current_time == datetime.timedelta(seconds=1):
                current_time = time
            else:
                multiple_data_logs.append(str(start_time) + ' ~ ' + str(current_time)\
                                          + ', for ' + str((current_time - start_time).seconds + 1) + 's\n')
                start_time = time
                current_time = time
        multiple_data_logs.append(str(start_time) + ' ~ ' + str(current_time)\
                                  + ', for ' + str((current_time - start_time).seconds + 1))

    if file_name == None:
        for log in no_data_logs:
            print(log, end='')
        print()
        for log in multiple_data_logs:
            print(log, end='')    
    else:
        with open(file_name, 'w') as f:
            for log in no_data_logs:
                f.write(log)
            f.write('\n')
            for log in multiple_data_logs:
                f.write(log) 
                
                
                
# Plot data by time.
def plot_data(data, columns, file_name=None):
    """
    data : data for plotting
    columns : (list of) name of column
    file_name : name of graph image file(.jpg format is recommaned), just show graph if file_name == None
    """
    plt.figure(figsize=(16, 12))
    if isinstance(columns, str):
        plt.plot(pd.to_datetime(data['TIMESTAMP']), data[columns])
    elif isinstance(columns, list) or isinstance(columns, tuple):
        for column in columns:
            plt.plot(pd.to_datetime(data['TIMESTAMP']), data[column])
            
    if file_name != None:
        plt.savefig(file_name, bbox_inches='tight')
    else:
        plt.show()
        
        
        
# Run a function repeatedly day by day.
def run_ft_daily(ft, data, first_date, last_date, use_date_input, *args, **kwargs):
    """
    ft : function to be run on data daily, the first argument should be data,
         arguments for ft are ft(data, *args, **kwargs)
    data : data for analysis by ft
    first_date : the first date when ft starts running, datatime.date or (year, month, day) format is required
    last_date : the last date when ft ends running, datatime.date or (year, month, day) format is required
    use_date_input : use each date as input for ft, the second argument should be date if use_date_input == True
    
    example) run_ft_daily(ft=plot_data,
                          data=bank_data,
                          first_date=(2021, 9, 30),
                          last_date=(2021, 11, 5),
                          use_date_input=False
                          ['BANK_DC_VOLT'],
                          file_name=None)
    """
    if isinstance(first_date, list) or isinstance(first_date, tuple):
        first_date = datetime.date(*first_date)
    if isinstance(last_date, list) or isinstance(last_date, tuple):
        last_date = datetime.date(*last_date)
        
    date = first_date
    if use_date_input:
        while date <= last_date:
            data_oneday = select_oneday_data(data, date)
            ft(data_oneday, date, *args, **kwargs)
            date = date + datetime.timedelta(days=1)
    else:
        while date <= last_date:
            data_oneday = select_oneday_data(data, date)
            ft(data_oneday, *args, **kwargs)
            date = date + datetime.timedelta(days=1)
            
            
            
# Remove microsecond in TIMESTAMP.
def remove_microsecond(data):
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP']).values.astype('datetime64[s]')
    return data



# Fill blank data in unobserved time.
def fill_unobserved_time_values(data, methods):
    """
    data : data with column 'TIMESTAMP'
    methods : method to fill NA data
              one of ['mean', 'linear'] or methods in pandas.DataFrame.fillna
              or list of methods that has the same length with the length of data columns except 'TIMESTAMP'
              
              'mean' -> fill NA with the average of both side of NA blocks
                        e.g. [0.0, NA, NA, NA, 2.0] -> [0.0, 1.0, 1.0, 1.0, 2.0]
              'linear' -> fill NA linearly
                          e.g. [0.0, NA, NA, NA, 2.0] -> [0.0, 0.5, 1.0, 1.5, 2.0]
                          
    example) fill_unobserved_time_values(bank_data, methods='linear')
             fill_unobserved_time_values(bank_data, methods=['mean', 'linear', 'ffill', 'mean'])
    """
    
    time = data['TIMESTAMP'].iloc[0]
    end_time = data['TIMESTAMP'].iloc[-1]
    time_series = [time]
    
    while time < end_time:
        time = time + datetime.timedelta(seconds=1)
        time_series.append(time)
        
    time_series = pd.Series(data=time_series, name='TIMESTAMP')
    filled_data = data.merge(time_series, how='right')
    
    if isinstance(methods, str):
        if methods == 'mean':
            for column in filled_data.columns:
                if column != 'TIMESTAMP':
                    na_data = filled_data[column].isna()
                    filled_data[column][na_data] = (filled_data[column].fillna(method='ffill')[na_data]\
                                                    + filled_data[column].fillna(method='bfill')[na_data]) / 2

        elif methods == 'linear':
            for column in filled_data.columns:
                if column != 'TIMESTAMP':
                    na_data = filled_data[column].isna()

                    index_series = pd.Series(data=filled_data.index.to_numpy())
                    _index_series = index_series.copy()
                    _index_series[na_data] = np.nan

                    index_delta = index_series[na_data] - _index_series.fillna(method='ffill')[na_data]
                    scale = _index_series.fillna(method='bfill')[na_data] - _index_series.fillna(method='ffill')[na_data]

                    na_ffill = filled_data[column].fillna(method='ffill')[na_data]
                    value_delta = filled_data[column].fillna(method='bfill')[na_data] - na_ffill

                    filled_data[column][na_data] = na_ffill + value_delta * index_delta / scale
                    
        else:
            filled_data.fillna(method=methods, inplace=True)
            
    else:
        columns = list(filled_data.columns)
        columns.remove('TIMESTAMP')
        for column, method in zip(columns, methods):
            if method == 'mean':
                na_data = filled_data[column].isna()
                filled_data[column][na_data] = (filled_data[column].fillna(method='ffill')[na_data]\
                                                + filled_data[column].fillna(method='bfill')[na_data]) / 2
                
            elif method == 'linear':
                na_data = filled_data[column].isna()

                index_series = pd.Series(data=filled_data.index.to_numpy())
                _index_series = index_series.copy()
                _index_series[na_data] = np.nan

                index_delta = index_series[na_data] - _index_series.fillna(method='ffill')[na_data]
                scale = _index_series.fillna(method='bfill')[na_data] - _index_series.fillna(method='ffill')[na_data]

                na_ffill = filled_data[column].fillna(method='ffill')[na_data]
                value_delta = filled_data[column].fillna(method='bfill')[na_data] - na_ffill

                filled_data[column][na_data] = na_ffill + value_delta * index_delta / scale
                
            else:
                filled_data[column].fillna(method=method, inplace=True)

    return filled_data



# Interpolate data linearly in equally-spaced timestamp. (unit : 1 second)
def equispaced_interpolation(data, numerical_columns, categorical_columns=None, first_time=None, last_time=None):
    """
    data : data with column 'TIMESTAMP'
    numerical_columns : list of numerical columns (except 'TIMESTAMP')
    categorical_columns : list of categorical columns
    first_time : the first time to interpolate, default is the first time in the given data without nanoseconds, type : datetime64[s]
    last_time : the last time to interpolate, default is the last time in the given data without nanoseconds, type : datetime64[s]
    
    example) equispaced_interpolation(bank_data,
                                      numerical_columns=['BANK_SOC', 'BANK_SOH', 'BANK_DC_VOLT', 'BANK_DC_CURRENT'],
                                      categorical_columns=['BATTERY_STATUS_FOR_CHARGE'])
                                      
             equispaced_interpolation(etc_data,
                                      ['SENSOR1_TEMPERATURE'],
                                      first_time=datetime.datetime(2021, 10, 1, 15, 0, 0),
                                      last_time=datetime.datetime(2021, 10, 2, 15, 0, 0))
    """
    base_time = pd.to_datetime(data['TIMESTAMP'].iloc[:1]).values.astype('datetime64[s]')[0]
    times = (pd.to_datetime(data['TIMESTAMP']).values - base_time).astype(int)
    
    ft_numerical = interpolate.interp1d(times, np.transpose(data[numerical_columns].values), fill_value='extrapolate')
    if categorical_columns != None:
        ft_categorical = interpolate.interp1d(times, np.transpose(data[categorical_columns].values), kind='next', fill_value='extrapolate')
    
    one_sec = 1000000000
    
    _first_time = 0 if first_time == None else (np.datetime64(first_time, 's') - base_time).astype(int) * one_sec
    _last_time = (pd.to_datetime(data['TIMESTAMP'].iloc[-1:]).values.astype('datetime64[s]')[0] - base_time).astype(int) * one_sec + 1\
                 if last_time == None else (np.datetime64(last_time, 's') - base_time).astype(int) * one_sec
    
    equispaced_times = np.arange(_first_time, _last_time, one_sec)
    interpolations_numerical = ft_numerical(equispaced_times)
    if categorical_columns != None:
        interpolations_categorical = ft_categorical(equispaced_times).astype(int)
        interpolations = np.concatenate((interpolations_numerical, interpolations_categorical), axis=0, dtype='object')
    else:
        interpolations = interpolations_numerical
    
    columns = [*numerical_columns, *categorical_columns] if categorical_columns != None else numerical_columns
    _data = pd.DataFrame(data=np.transpose(interpolations), columns=columns)
    _data['TIMESTAMP'] = (equispaced_times // one_sec).astype('timedelta64[s]') + base_time
    
    return _data



# Split and daily data by BATTERY_STATUS_FOR_CHARGE
def split_daily_data(data):
    """
    data : daily data with column 'BATTERY_STATUS_FOR_CHARGE'
           daily data should have the process with 'charge -> idle -> discharge'

    output : charge state data, idle state data when battery is charged, discharge state data
    """
    charge = data[(data['BATTERY_STATUS_FOR_CHARGE'] == 2)]
    charge_start = charge.index[0]

    discharge = data[(data['BATTERY_STATUS_FOR_CHARGE'] == 3)]
    discharge_end = discharge.index[-1]

    idle = data.loc[charge_start:discharge_end][(data['BATTERY_STATUS_FOR_CHARGE'] == 1)]

    return charge, idle, discharge
