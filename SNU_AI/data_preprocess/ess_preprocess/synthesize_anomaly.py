import numpy as np


# Configurations for anomaly synthesis
N_ANOMALY_TYPES = 4
REPLACING_RATE = (0.0005, 0.1)  # min-max rate of the length of replacing interval for anomalies

# anomaly synthesis options
MAX_EXTERNAL_INTERVAL_RATE = 0.7
MAX_UNIFORM_VALUE_DIFFERENCE = 0.1
MIN_PEAK_ERROR = 0.1
WHITE_NOISE_LEVEL = 0.003



# Soft replacement
def soft_replacement(data, target_index, interval_len, abnormal_column, external_interval):
    target_interval = data[target_index:target_index+interval_len, abnormal_column].copy()
    syn_data = data.copy()
    
    weights = np.concatenate((np.linspace(0, MAX_EXTERNAL_INTERVAL_RATE, num=interval_len//2),
                              np.linspace(MAX_EXTERNAL_INTERVAL_RATE, 0, num=(interval_len+1)//2)), axis=None)
    syn_data[target_index:target_index+interval_len, abnormal_column] = weights[:, None] * external_interval\
                                                                        + (1 - weights[:, None]) * target_interval
    
    anomaly_label = np.zeros((len(data), 1))
    anomaly_label[target_index:target_index+interval_len] = 1
    
    return syn_data, anomaly_label


# Uniform replacement
def uniform_replacement(data, target_index, interval_len, abnormal_column):
    syn_data = data.copy()
    mean_values = syn_data[target_index:target_index+interval_len, abnormal_column].mean(axis=0)
    syn_data[target_index:target_index+interval_len, abnormal_column]\
        = np.random.uniform(np.maximum(mean_values-MAX_UNIFORM_VALUE_DIFFERENCE, 0),
                            np.minimum(mean_values+MAX_UNIFORM_VALUE_DIFFERENCE, 1))[None, :]
    
    anomaly_label = np.zeros((len(data), 1))
    anomaly_label[target_index:target_index+interval_len] = 1
    
    return syn_data, anomaly_label


# Peak noise
def peak_noise(data, target_index, interval_len, abnormal_column):
    syn_data = data.copy()
    peak_indices = np.random.randint(target_index, target_index+interval_len,
                                     size=len(abnormal_column[abnormal_column]))
    peak_values = syn_data[peak_indices, abnormal_column].copy()
    
    peak_errors = np.random.uniform(np.minimum(0, MIN_PEAK_ERROR-peak_values), np.maximum(0, 1-peak_values-MIN_PEAK_ERROR))
    peak_values = peak_values + peak_errors + ((peak_errors > 0).astype(int) * 2 - 1) * MIN_PEAK_ERROR
    syn_data[peak_indices, abnormal_column] = peak_values
    
    anomaly_label = np.zeros((len(data), 1))
    anomaly_label[peak_indices] = 1
    
    return syn_data, anomaly_label


# White noise
def white_noise(data, target_index, interval_len, abnormal_column):
    syn_data = data.copy()
    noised_data = syn_data[target_index:target_index+interval_len, abnormal_column]\
                  + np.random.randn(interval_len, len(abnormal_column[abnormal_column])) * WHITE_NOISE_LEVEL
    
    noised_data[noised_data > 1] = 1
    noised_data[noised_data < 0] = 0
    syn_data[target_index:target_index+interval_len, abnormal_column] = noised_data
    
    anomaly_label = np.zeros((len(data), 1))
    anomaly_label[target_index:target_index+interval_len] = 1
    
    return syn_data, anomaly_label