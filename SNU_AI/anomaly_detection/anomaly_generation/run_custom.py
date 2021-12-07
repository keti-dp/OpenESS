import json
import pandas as pd
from univariate import UnivariateDataGenerator

with open('./config.json', 'r') as f: 
    config = json.load(f)

df = pd.read_csv('sample.csv')
ts = df['value'].values

univariate_data = UnivariateDataGenerator()
univariate_data.load_timeseries(ts)

univariate_data.collective_shapelet_outliers(
    ratio=config['collective_shapelet']['ratio'], 
    radius=config['collective_shapelet']['radius'], 
    coef=config['collective_shapelet']['coef'], 
    noise_amp=config['collective_shapelet']['noise_amp'], 
    base=config['collective_shapelet']['base']
) #2
# univariate_data.collective_seasonal_outliers(
#     ratio=config['collective_seasonal']['ratio'], 
#     factor=config['collective_seasonal']['factor'], 
#     radius=config['collective_seasonal']['radius']
# ) #3
univariate_data.collective_trend_outliers(
    ratio=config['collective_trend']['ratio'], 
    factor=config['collective_trend']['factor'], 
    radius=config['collective_trend']['radius']
) #4

univariate_data.point_global_outliers(
    ratio=config['point_global']['ratio'], 
    factor=config['point_global']['factor'],
    factor_noise_amp=config['point_global']['factor_noise_amp']
) #0
univariate_data.point_contextual_outliers(
    ratio=config['point_contextual']['ratio'], 
    factor=config['point_contextual']['factor'],  
    radius=config['point_contextual']['radius'],  
    factor_noise_amp=config['point_contextual']['factor_noise_amp']
) #1

univariate_data.plot()