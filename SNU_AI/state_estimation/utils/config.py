import os

# Edit DATA_DIR's below to your own dataset directory.
RAW_DATA_DIR = ''  # raw nasa dataset directory
PROCESSED_DATA_DIR = ''  # processed nasa dataset directory

PROJECT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

BATTERY_NAME = ['RW{:02}.parquet'.format(i) for i in range(1, 13)]
CYCLE_INDEX_INFO = PROJECT_DIR + '/data/cycle_index_info.npz'