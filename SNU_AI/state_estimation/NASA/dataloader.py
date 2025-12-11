from typing import *
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


INPUT_DATA_PATH = './encoder_data/'
OUTPUT_DATA_PATH = './decoder_data/'
# FILE_NUM_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20]
# TEST_NUM_LIST = [1, 3, 13, 17]
# FILE_NUM_LIST = [3, 4, 5, 6]
# TEST_NUM_LIST = [6]



class CustomDataset(Dataset):
    def __init__(
        self,
        mode: str,
        input_process_type: str,
        output_type: str,
        train_battery: list,
        test_battery: list
    ):
        super().__init__()
        assert mode in ['train', 'test']
        if mode == 'train':
            # self.file_num_list = [file_num for file_num in FILE_NUM_LIST if file_num not in TEST_NUM_LIST]
            self.file_num_list = train_battery
        elif mode == 'test':
            # self.file_num_list = TEST_NUM_LIST
            self.file_num_list = test_battery
        self._load_data(input_process_type, output_type)

    def __len__(self):
        return len(self.soh)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.soh[idx], self.drive[idx, :], self.ri_curve[idx, :]

    def _load_data(self, process_type: str = 'cut64', output_type: str = 'vec') -> None:
        assert process_type.startswith(('cut', 'mask'))
        assert output_type in ['vec', 'poly'], f'Output type should be `vec` (vector) or `poly` (polynomial coefficients)!'
        self.soh = []
        self.drive = []
        self.ri_curve = []
        for file_num in self.file_num_list:
            vi2soh_data = np.load(os.path.join(INPUT_DATA_PATH, f'RW{file_num:02d}_rw_dis_V_I_SOH.npz'), allow_pickle=True)['data']
            soh2ri_data = np.load(os.path.join(OUTPUT_DATA_PATH, f'RW{file_num:02d}_ref_dis_SOC_Ri.npz'))['data']
            min_soh, max_soh = soh2ri_data[0, 0], soh2ri_data[-1, 0] # Min, Max SOH for soh2ri data
            for i in range(len(vi2soh_data)):

                # (1) Get SOH
                soh = round(vi2soh_data[i][0], 4)

                # (2) Match SOH & Get the `corresponding` ri_curve
                soh_idx = int((soh - min_soh) / (max_soh - min_soh) * soh2ri_data.shape[0])
                if soh_idx < 0 or soh_idx >= soh2ri_data.shape[0]: # no soh matched for soh2ri data
                    continue

                # (3) Get input cycle (cut or mask)
                drive_cycle = vi2soh_data[i][1]
                if process_type.startswith('cut'): # For MLP, CNN, RNN
                    cut_len = int(process_type.split('cut')[1])
                    if len(drive_cycle) >= cut_len:
                        start = np.random.randint(low=0, high=len(drive_cycle)-cut_len+1)
                        drive_cycle = drive_cycle[start:start+cut_len, :]
                    else:
                        continue
                elif process_type.startswith('mask'): # For Transformers
                    max_len = int(process_type.split('mask')[1])
                    if len(drive_cycle) >= max_len:
                        start = np.random.randint(low=0, high=len(drive_cycle)-max_len+1)
                        drive_cycle = drive_cycle[start:start+max_len, :]
                        drive_cycle = np.concatenate([drive_cycle, np.zeros((len(drive_cycle), 1))], axis=1)
                    else:
                        drive_cycle = np.concatenate([drive_cycle, np.zeros((len(drive_cycle), 1))], axis=1)
                        mask_len = max_len - len(drive_cycle)
                        mask = np.concatenate([np.zeros((mask_len, 2)), np.ones((mask_len, 1))], axis=1)
                        drive_cycle = np.concatenate([drive_cycle, mask], axis=0)
                else:
                    raise ValueError(f'Process type should be `cut` type or `mask` type!')

                # (4) Append data
                self.soh.append(soh)
                self.drive.append(drive_cycle)
                self.ri_curve.append(soh2ri_data[soh_idx, 1:])

        self.soh = np.array(self.soh, dtype=np.float32)
        self.drive = np.stack(self.drive, axis=0).astype(np.float32)
        self.drive = normalize_vi(self.drive)
        self.ri_curve = np.stack(self.ri_curve, axis=0).astype(np.float32)
        assert len(self.soh) == len(self.drive)
        assert len(self.soh) == len(self.ri_curve)


def get_dataloader(batch_size: int, input_process_type: str, output_type: str, train_battery: list, test_battery: list) -> Tuple[DataLoader, DataLoader]:
    train_dset = CustomDataset('train', input_process_type, output_type, train_battery, test_battery)
    test_dset = CustomDataset('test', input_process_type, output_type, train_battery, test_battery)
    train_loader = DataLoader(train_dset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dset, batch_size, shuffle=False)
    return train_loader, test_loader


def normalize_vi(vi: np.ndarray) -> np.ndarray:
    new_vi = np.empty_like(vi)
    new_vi[:, :, 0] = vi[:, :, 0] - 3.2
    new_vi[:, :, 1] = vi[:, :, 1] / 5
    return new_vi


def denormalize_vi(vi: np.ndarray) -> np.ndarray:
    new_vi = np.empty_like(vi)
    new_vi[:, :, 0] = vi[:, :, 0] + 3.2
    new_vi[:, :, 1] = vi[:, :, 1] * 5
    return new_vi


if __name__ == '__main__':
    print('[CUT64 - example (for MLP, CNN, RNN)]')
    train_loader, test_loader = get_dataloader(32, 'cut64', 'vec')
    for soh, drive, ri_curve in train_loader:
        print('SOH: ', soh.shape)
        print('DRIVE: ', drive.shape)
        print('RI_curve: ', ri_curve.shape)
        break

    print('[MASK512 - example (for TRANSFORMER)]')
    train_loader, test_loader = get_dataloader(32, 'mask512', 'vec')
    for soh, drive, ri_curve in train_loader:
        print('SOH: ', soh.shape)
        print('DRIVE: ', drive.shape)
        print('RI_curve: ', ri_curve.shape)
        break