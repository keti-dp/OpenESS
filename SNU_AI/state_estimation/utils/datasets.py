import os, sys
sys.path.append(os.path.dirname(__file__))

import copy
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, Sampler, DataLoader

import config



# NASA dataset
class NasaDataset(Dataset):
    def __init__(self, set_type, target, columns):
        super(NasaDataset, self).__init__()
        self.set_type = set_type
        self.columns = columns
        
        if set_type == 'train':
            self.usable_id = [2, 3, 4]
        elif set_type == 'test':
            self.usable_id = [5]
        elif set_type == 'valid':
            self.usable_id = [5]
        else:
            self.usable_id = list(map(int, set_type.split(',')))
        self.usable_id = np.array(self.usable_id)
        
        dataset_dir = config.DATA_DIR
        battery_name = config.BATTERY_NAME
        
        self.dataframe = {}
        for _id in self.usable_id:
            self.dataframe[_id] = pd.read_parquet(dataset_dir + battery_name[_id])[columns]
        
        with np.load(config.CYCLE_INDEX_INFO) as data_infos:
            self.battery_id = data_infos['battery_id']
            self.cycle = data_infos['cycle']
            self.first_index = data_infos['first_index']
            self.last_index = data_infos['last_index']
            self.target = data_infos[target]
            
        self.usable_index = np.argwhere(self.battery_id - self.usable_id[:, np.newaxis] == 0)[:, 1]
        self.length = len(self.usable_index)
        
    def get_data_item(self, item, copy_item=False):
        if copy_item:
            return copy.deepcopy(getattr(self, item))
        else:
            return getattr(self, item)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        """
        index : (meta_index, first_index, last_index)
        """
        item = {}
        battery_id = self.battery_id[index[0]]
        df = self.dataframe[battery_id]
        
        data = torch.Tensor(df.iloc[index[1]:index[2], :].values)
        if 'dt' in self.columns:
            data[0, -1] = 0  # first dt = 0 if dt is in columns
        item['data'] = data
        
        item['target'] = self.target[index[0]]
        item['cycle'] = self.cycle[index[0]]
        item['battery_id'] = battery_id
        item['first_index'] = index[1]
        item['last_index'] = index[2]
        
        return item
    
    
    
# NASA dataset sampler
class NasaDatasetSampler(Sampler):
    def __init__(self, data_source, sliding_window, shuffle):
        """
        data_source : NasaDataset
        sliding_window : number of history vertors for input for RNN
        shuffle : whether to shuffle data indices
        """
        self.sliding_window = sliding_window
        self.shuffle = shuffle
        
        self.usable_index = data_source.get_data_item('usable_index', copy_item=True)
        self.first_index = data_source.get_data_item('first_index')
        self.last_index = data_source.get_data_item('last_index', copy_item=True) - sliding_window + 1
        
        invalid_meta_index = np.argwhere(self.last_index <= self.first_index)
        self.usable_index = np.delete(self.usable_index, np.argwhere(self.usable_index - invalid_meta_index == 0)[:, 1])
        self.length = len(self.usable_index)
        
    def __len__(self):
        return self.length
    
    def __iter__(self):
        meta_index = np.random.permutation(self.usable_index) if self.shuffle else self.usable_index
        df_index = np.random.randint(self.first_index[meta_index], self.last_index[meta_index])
        yield from list(zip(meta_index, df_index, df_index + self.sliding_window))
        
        
        
# Collate function for NASA dataset
def nasa_dataset_collate_fn(batch):
    _batch = {}
    data_list = []
    target_list = []
    cycle_list = []
    battery_id_list = []
    first_indices = []
    last_indices = []

    for item in batch:
        data_list.append(item['data'])
        target_list.append(item['target'])
        cycle_list.append(item['cycle'])
        battery_id_list.append(item['battery_id'])
        first_indices.append(item['first_index'])
        last_indices.append(item['last_index'])
        
    _batch['data'] = torch.stack(data_list)
    _batch['target'] = torch.Tensor(target_list)
    _batch['cycle'] = cycle_list
    _batch['battery_id'] = battery_id_list
    _batch['first_index'] = first_indices
    _batch['last_index'] = last_indices
    
    return _batch



# Get a data loader for NASA dataset.
def get_nasa_dataset_loader(set_type='train',
                            target='soh',
                            columns=['V', 'I', 'T', 'dt'],
                            batch_size=32,
                            sliding_window=32,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True):
    
    dataset = NasaDataset(set_type=set_type, target=target, columns=columns)
    sampler = NasaDatasetSampler(dataset, sliding_window=sliding_window, shuffle=shuffle)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, pin_memory=pin_memory, drop_last=drop_last, collate_fn=nasa_dataset_collate_fn)