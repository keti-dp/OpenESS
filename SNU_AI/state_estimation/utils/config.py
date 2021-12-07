import os

# Edit DATA_DIR to your own dataset directory.
DATA_DIR = ''

PROJECT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

BATTERY_NAME = ['RW{:02}.parquet'.format(i) for i in range(1, 13)]
CYCLE_INDEX_INFO = PROJECT_DIR + '/data/cycle_index_info.npz'