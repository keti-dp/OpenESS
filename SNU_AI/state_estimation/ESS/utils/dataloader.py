import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import random
import os
from utils.preprocess import *

import time


class ESSDataset(Dataset):
    def __init__(self, config, data, transforms=None, mode=None):
        super().__init__()
        self.config = config
        self.data = data
        self.transforms = transforms
        self.sample_length = config['seconds']
        self.mode = mode

        self.input_cols = config['input_cols']
        self.target_col = config['target_col']
        
    # def __getitem__(self, index):
    #     # if self.mode == 'test':
    #     #     torch.manual_seed(self.config['seed'])
    #     sample = self.data[index]
    #     sample = self.cut_window(sample)
    #     sample.rename(columns=COL2NAME_SIONYU, inplace=True)
    #     input, target = sample[self.input_cols], sample[self.target_col]
    #     input, target = torch.tensor(input.values), torch.tensor(target.values)
    #     input, target = torch.transpose(input, 0,1), torch.transpose(target, 0,1)  # channel(cols) x length(86400)
    #     if self.transforms:
    #         input_transform, target_transform = self.transforms
    #         input = input_transform(input)
    #         target = target_transform(target)
    #     return input.type(torch.float32), target.mean(axis=-1).type(torch.float32) # datatype 수정, 1분마다 soc 찍히도록 평균 냄
    
    def __getitem__(self, index):
        # if self.mode == 'test':
        #     torch.manual_seed(self.config['seed'])
        inputs = []
        targets = []
        for _ in range(self.config['data_per_oneday']):
            sample = self.data[index]
            sample = self.cut_window(sample)
            sample.rename(columns=COL2NAME_SIONYU, inplace=True)
            input, target = sample[self.input_cols], sample[self.target_col]
            input, target = torch.tensor(input.values), torch.tensor(target.values)
            input, target = torch.transpose(input, 0,1), torch.transpose(target, 0,1)  # channel(cols) x length(86400)
            if self.transforms:
                input_transform, target_transform = self.transforms
                input = input_transform(input)
                target = target_transform(target)
            inputs.append(input.type(torch.float32))
            targets.append(target.mean(axis=-1).type(torch.float32))
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        return inputs, targets
    
    
    def __len__(self):
        return len(self.data)
    

    def cut_window(self, sample):
        BATTERY_STATUS_FOR_CHARGE = self.config['BATTERY_STATUS_FOR_CHARGE']
        rest = True
        while rest :
            start_index = torch.randint(0,len(sample) - self.sample_length + 1,(1,))
            start_index = start_index.item()
            # if (sample[start_index:start_index + self.sample_length]['BATTERY_STATUS_FOR_CHARGE']==1).values.sum()==0 : ##모두 False, 즉 rest인 구간이 없을 때.
            if (sample[start_index:start_index + self.sample_length]['BATTERY_STATUS_FOR_CHARGE']==BATTERY_STATUS_FOR_CHARGE).all():
                rest = False
        window_data = sample[start_index:start_index + self.sample_length]
        return window_data


class ESSDataLoader(DataLoader):
    def __init__(self, config, dataset, mode=None, *args, **kwargs):
    # def __init__(self, dataset, config, *args, cols=None, seconds=None, **kwargs):
        self.mode = mode
        if self.mode == 'test':
            self.batch_size = config['test_batch_size']
            self.shuffle = False
        else:
            self.batch_size = config['train_batch_size']
            self.shuffle = True


        super(ESSDataLoader,self).__init__(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=config['num_workers'], drop_last=config['drop_last'], *args, **kwargs)
        self.seconds = config['seconds']

    
    

def change_col_name(site, df):
    if site == 'sionyu':
        for key, value in COL2NAME_SIONYU.items():
            df.rename(columns={key:value}, inplace=True)
    elif site == 'panli':
        for key, value in COL2NAME_PANLI.items():
            df.rename(columns={key:value}, inplace=True)   
    return df

def get_raw_cols(site):
    if site == 'sionyu':
        raw_cols = RAW_COLUMNS_SIONYU
    elif site == 'panli':
        raw_cols = RAW_COLUMNS_PANLI
    return raw_cols

def make_oneday_data(site, debug):
    path = f'/data/ess/data/incell/state_estimation_data/{site}/ocv_labeled'
    data_path = f'/data/ess/data/incell/state_estimation_data/{site}/ocv_labeled/oneday_data'

    # load
    if os.path.exists( data_path ):
        parquets = [os.path.join(data_path, file) for file in os.listdir(data_path)]
        parquets.sort()
        oneday_data = [pd.read_parquet(parquet) for parquet in parquets]


    # make
    else:
        os.makedirs(data_path, exist_ok=False)
        oneday_data = []
        parquet_list = [file for file in os.listdir(path) if 'normalized' in file ]

        for parquet_file in parquet_list:
            df = pd.read_parquet( os.path.join(path, parquet_file) )
            raw_cols = get_raw_cols(site)
            df = df[raw_cols]
            dates = pd.to_datetime(df['TIMESTAMP'], utc=True).dt.tz_convert('Asia/Seoul').dt.date.unique()
            
            for date in dates:
                file_name = str(date) + parquet_file[13:]
                data = select_oneday_data(df,date=date)
                data.to_parquet( os.path.join(data_path, file_name) )
                oneday_data.append(data)

    if debug:
        oneday_data = oneday_data[:2]

    return oneday_data

def split(oneday_data, split_ratio):
    split_ratio = np.array(split_ratio)
    split_ratio = split_ratio / sum(split_ratio)
    a, b, c = split_ratio
    L = len(oneday_data)
    a, b, c = int(a*L), int(b*L), int(c*L)
    return oneday_data[:a], oneday_data[a:a+b], oneday_data[a+b:]


def build_loader(config):

    print('Start building loader')
    start = time.time()
    oneday_data = make_oneday_data(config['site'], config['debug'])


    if config['task'] == 'estimation':
        random.shuffle(oneday_data)
        train, val, test = split(oneday_data, config['split_ratio'])
    elif config['task'] == 'prediction':
        train, val, test = split(oneday_data, config['split_ratio'])
    elif config['task'] == 'custom':
        # 웨부부부붸베베붸베 채워
        print('채워임마')


    # train_transform = transforms.Compose([
    #     Normalize()
    #     ])
    # test_transform = transforms.Compose([
    #     SOCNormalize()
    #     ])
    
    input_transform = transforms.Compose([
        # Normalize()
        ])
    target_transform = transforms.Compose([
        # SOCNormalize()
        ])
    input_target_transform = [input_transform, target_transform]
    
    # train, val, test = ESSDataset(train, transforms=train_transform), ESSDataset(val, transforms=test_transform), ESSDataset(test, transforms=test_transform)
    train, val, test = ESSDataset(config, train, transforms=input_target_transform), ESSDataset(config, val, transforms=input_target_transform, mode='test'), ESSDataset(config, test, transforms=input_target_transform, mode='test')
    train_loader, val_loader, test_loader = ESSDataLoader(config, train), ESSDataLoader(config, val, mode='test'), ESSDataLoader(config, test, mode='test')

    print(f'It takes {time.time()-start:.2f} seconds for dataset&dataloader')

    return train_loader, val_loader, test_loader




###################################################################################################
# transforms

class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        col_max = sample.max(axis=1).values.unsqueeze(1)
        col_min = sample.min(axis=1).values.unsqueeze(1)
        return (sample - col_min) / (col_max - col_min + 1e-8)
    
class SOCNormalize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample / 100
    
    
class Select_Columns(object):
    def __init__(self, site, cols):

        if site == 'sionyu':
            self.cols = [NAME2COL_SIONYU[col] for col in cols]
        elif site == 'panli':
            self.cols = [NAME2COL_PANLI[col] for col in cols]

    def __call__(self, sample):
        return sample[self.cols]
    

class DF2Tensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return torch.tensor(sample.values)
    

class Gaussian_Perturbation(object):
    def __init__(self, cols):
        self.cols = cols

    def __call__(self, sample):
        return sample
    