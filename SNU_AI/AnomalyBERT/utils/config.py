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
DATASET_DIR = 'datasets/processed'
LOG_DIR = 'logs/'
DATA_PROPERTY_DIR = 'data/'


DATASET_LIST = ['ESS_sionyu', 'ESS_panli_bank1', 'ESS_panli_bank2']

TRAIN_DATASET = {}
TEST_DATASET = {}
TEST_LABEL = {}

for data_name in DATASET_LIST:
    TRAIN_DATASET[data_name] = os.path.join(DATASET_DIR, data_name + '_train.npy')
    TEST_DATASET[data_name] = os.path.join(DATASET_DIR, data_name + '_test.npy')
    TEST_LABEL[data_name] = os.path.join(DATASET_DIR, data_name + '_test_labels.npy')


DATA_DIVISION = {}

DEFAULT_DIVISION = {'ESS_sionyu' : 'total',
                    'ESS_panli_bank1' : 'total',
                    'ESS_panli_bank2' : 'total',
                   }


NUMERICAL_COLUMNS = {'ESS_sionyu' : range(0,5),
                     'ESS_panli_bank1' : range(0,5),
                     'ESS_panli_bank2' : range(0,5)
                    }

CATEGORICAL_COLUMNS = {'ESS_sionyu' : tuple(),
                       'ESS_panli_bank1' : tuple(),
                       'ESS_panli_bank2' : tuple()
                      }

IGNORED_COLUMNS = {}

