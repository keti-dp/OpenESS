"""
AnomalyBERT
################################################

Reference:
    Yungi Jeong et al. "AnomalyBERT: Self-Supervised Transformer for Time Series Anomaly Detection using Data Degradation Scheme" in ICLR Workshop, "Machine Learning for Internet of Things(IoT): Datasets, Perception, and Understanding" 2023.

Reference:
    https://github.com/Jhryu30/AnomalyBERT
"""

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DATASET_DIR = '/data/ess/data/incell/year45_preprocessed/0612_temperature/integrated/'
# DATASET_DIR = '/data/ess/data/incell/year45_preprocessed/integrated/'
LOG_DIR = '/data/ess/output/Anomaly_Detection/logs/'

DATASET_LIST = ['ESS_sionyu', 'ESS_panli', 'ESS_gold', 'ESS_white']

TRAIN_DATASET = {}
TEST_DATASET = {}

for data_name in DATASET_LIST:
    # TRAIN_DATASET[data_name] = os.path.join(DATASET_DIR, data_name + '_train.npy')
    TRAIN_DATASET[data_name] = os.path.join(DATASET_DIR, data_name + '_train.npy')
    TEST_DATASET[data_name] = os.path.join(DATASET_DIR, data_name + '_test.npy')

# TEST_DATASET['testbed'] = '/home/parkis/AnomalyDetection/anomalybert4ESS/AnomalyBERT/data_to_try.npy' # data .npy file path
TEST_DATASET['testbed'] = '' # data .npy file path