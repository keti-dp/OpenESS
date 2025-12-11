import os
import numpy as np
import torch
from torch.utils.data import Dataset
import utils.config as config

class ESS_dataset(Dataset):
    def __init__(self, options, seed=None, is_train=True):
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.options = options
        data_options = options.data
        try:
            data_path = config.TRAIN_DATASET[options.dataset] if is_train else config.TEST_DATASET[options.dataset]
        except KeyError:
            data_path = options.dataset
            options.dataset = 'other'
         
        data = np.load(data_path).astype(np.float32) # time x (1 + columns + 1)
        if data.shape[1] == 5:
            data = np.concatenate([np.zeros((len(data), 1)), data, np.zeros((len(data), 1))], axis=1)

        self.voltage_gap_seq = data[:,-1]
        self.charge_status = data[:,0]
        self.data = data[:,1:-1]
        self.time_len, self.column_len = self.data.shape
        self.data_seq_len = options.network.n_features * options.network.patch_size
        if self.data_seq_len >= self.time_len:
            print("Interpolate data.")
            x = np.arange(self.time_len)
            x_new = np.linspace(0, self.time_len-1, 86400)
            self.data = np.stack([np.interp(x_new, x, self.data[:,i]) for i in range(5)], axis=1)
            self.volatge_gap_seq = np.interp(x_new, x, self.voltage_gap_seq)
            self.time_len = 86400

        self.valid_index_list = np.arange(len(self.data) - self.data_seq_len)
        self.numerical_column = np.arange(self.column_len)
        
        self.replacing_rate_max = data_options.replacing_rate_max
        self.replacing_weight = data_options.replacing_weight

        # anomaly probs
        self.voltage_gap_prob = 1 - data_options.voltage_gap
        self.temperature_anomaly =  self.voltage_gap_prob - data_options.temperature_anomaly
        self.soft_replacing_prob = self.temperature_anomaly - data_options.soft_replacing
        self.uniform_replacing_prob = self.soft_replacing_prob - data_options.uniform_replacing
        self.peak_noising_prob = self.uniform_replacing_prob - data_options.peak_noising
        self.white_noising_prob = self.peak_noising_prob - data_options.white_noising
        
        # flip
        flip_replacing_interval = data_options.flip_replacing_interval.lower()
        if flip_replacing_interval == 'all':
            self.vertical_flip = True
            self.horizontal_flip = True
        elif flip_replacing_interval == 'vertical':
            self.vertical_flip = True
            self.horizontal_flip = False
        elif flip_replacing_interval == 'horizontal':
            self.vertical_flip = False
            self.horizontal_flip = True
        elif flip_replacing_interval == 'none':
            self.vertical_flip = False
            self.horizontal_flip = False

    def __len__(self):
        if self.seed is None:
            return self.options.batch_size * (self.options.max_steps + 1)
        else:
            return len(self.data)
    
    def __getitem__(self, index):
        assert self.seed is None, 'It is not a train dataset.'
        first_index = np.random.choice(self.valid_index_list) # model input으로 들어갈 수 있는 index 중 하나 선택
        return self.get_anomaly_data(index, first_index)
    
    def get_anomaly_data(self, index, first_index):
        x = torch.tensor(self.data[first_index:first_index+self.data_seq_len].copy()).float() # size = (data_seq_len, column_len)
        charge_status = torch.tensor(self.charge_status[first_index:first_index+self.data_seq_len].copy()) # size = (data_seq_len, column_len)
        x_true = x.clone()
        x_anomaly = torch.zeros(self.data_seq_len)

        replacing_length = np.random.randint(int(self.data_seq_len*self.replacing_rate_max/10), int(self.data_seq_len*self.replacing_rate_max)) # replacing length 선택

        target_index = np.random.randint(0, self.data_seq_len-replacing_length+1) # size = (1,), model input 중 replacing 구간 및 처음 index 선택
        replacing_type = np.random.uniform(0., 1.) # replacing type을 정하기 위한 변수
        replacing_dim_numerical = np.random.uniform(0., 1., size=self.column_len) # size = (column_len,)
        replacing_dim_numerical = (replacing_dim_numerical - np.maximum(np.min(replacing_dim_numerical), 0.3)) <= 0.001 # 무조건 True 한개 이상 존재, replacing column 선택 위한 변수

        if replacing_length > 0:
            is_replace = True

            # voltage_gap anomaly는 전 구간을 사용하므로 우선적으로 처리함
            if replacing_type > self.voltage_gap_prob:
                # Voltage Gap을 반영한 New Voltage 계산 후, 기존 Voltage와 비교하여 New Voltage Gap 계산함
                newV = x[:,0] + torch.Tensor(self.voltage_gap_seq[target_index : target_index+self.data_seq_len])
                x[:,-1] += torch.abs(newV-x[:,0])
                x_anomaly[:] = 1
            
            elif replacing_type > self.temperature_anomaly:
                x, anomaly_start, peak, anomaly_end = self._temperature_anomaly(x, charge_status)
                if anomaly_start is not None:
                    x_anomaly[anomaly_start:anomaly_end] = 1

            else:
                _x = x[target_index:target_index+replacing_length].clone().transpose(0, 1) # size = (column_len, replacing_len)
                replacing_number = sum(replacing_dim_numerical) # replacing column 개수
                target_column_numerical = self.numerical_column[replacing_dim_numerical] # replacing column 선택

                if replacing_type > self.soft_replacing_prob:
                    _x[target_column_numerical] = self._soft_replacing(_x[target_column_numerical], num=replacing_number, length=replacing_length)
                    x_anomaly[target_index:target_index+replacing_length] = 1

                elif replacing_type > self.uniform_replacing_prob:
                    _x[target_column_numerical] = self._uniform_replacing(num=replacing_number)
                    x_anomaly[target_index:target_index+replacing_length] = 1

                elif replacing_type > self.peak_noising_prob:
                    peak_value, peak_index = self._peak_noising(_x[target_column_numerical], num=replacing_number, length=replacing_length)
                    _x[target_column_numerical, peak_index] = peak_value

                    peak_index += target_index
                    target_first = np.maximum(0, peak_index - self.options.network.patch_size) # patch 안에 존재하는지 확인
                    target_last = peak_index + self.options.network.patch_size + 1
                    x_anomaly[target_first:target_last] = 1

                elif replacing_type > self.white_noising_prob:
                    _x[target_column_numerical] = self._white_noising(_x[target_column_numerical], num=replacing_number, length=replacing_length)
                    x_anomaly[target_index:target_index+replacing_length] = 1

                else:
                    is_replace = False

                if is_replace:
                    x[target_index:target_index+replacing_length] = _x.transpose(0, 1)
                         
        return x, x_anomaly, x_true

    def _temperature_anomaly(self, x, charge_status, intensity=None):
        """
        Applied on only charge status. Especially tail of charge status.
        """
        def has_contiguous_ones_only(one_indices):
            
            if len(one_indices) == 0:
                return True  # 1이 하나도 없으면 OK

            # 시작부터 끝까지가 연속된 숫자여야 함
            expected = np.arange(one_indices[0], one_indices[-1] + 1)
            return np.array_equal(one_indices, expected)
        
        
        one_indices = np.where(charge_status == 1)[0]
        charge_length = len(one_indices)

        if charge_length <= 1000 or (not has_contiguous_ones_only(one_indices)):
            # Too short or not continuous charge status
            return x, None, None, None
        else:
            # get indexes
            anomaly_length_ratio = np.random.uniform(low=0.2, high=0.5, size=1)
            anomaly_start = int(one_indices[-1] - charge_length * anomaly_length_ratio)
            peak = one_indices[-1]
            anomaly_end = min(int(peak + 0.5 * charge_length * anomaly_length_ratio), x.shape[0])
            
            if intensity is None: intensity = np.random.uniform(low=1.1, high=1.2, size=1).item()
            
            x[anomaly_start: peak, 3] *= torch.linspace(1, intensity, steps=peak-anomaly_start)
            x[peak: anomaly_end, 3] *= torch.linspace(intensity, 1, steps=anomaly_end-peak)
            
            return x, anomaly_start, peak, anomaly_end


    def _soft_replacing(self, x, num, length):
        replacing_index = np.random.randint(0, self.time_len-length+1, size=num) # size = (replacing_num,)
        _x = []
        col_num = np.random.choice(self.numerical_column, size=num)
        flip = np.random.randint(0, 2, size=(num, 2)) > 0.5
        for _col, _rep, _flip in zip(col_num, replacing_index, flip):
            random_interval = self.data[_rep:_rep+length, _col].copy()
            if self.horizontal_flip and _flip[0]:
                random_interval = random_interval[::-1].copy()
            if self.vertical_flip and _flip[1]:
                random_interval = 1 - random_interval
            _x.append(torch.from_numpy(random_interval).float())
        
        _x = torch.stack(_x)
        warmup_len = length//10
        weights = torch.concat((torch.linspace(0, self.replacing_weight, steps=warmup_len),
                                torch.full(size=(length-2*warmup_len,), fill_value=self.replacing_weight),
                                torch.linspace(self.replacing_weight, 0, steps=warmup_len)), dim=0).float().unsqueeze(0)

        return _x * weights + x * (1-weights)

    def _uniform_replacing(self, num):
        return torch.rand(size=(num, 1)).float()
    
    def _peak_noising(self, x, num, length):
        peak_index = np.random.randint(0, length)
        peak_value = (x[:,peak_index] < 0.5).float()
        peak_value = peak_value + (0.1 * (1 - 2 * peak_value)) * torch.rand(size=(num,))

        return peak_value, peak_index

    def _white_noising(self, x, num, length):
        return (x+torch.normal(mean=0, std=0.003, size=(num, length))).clamp(min=0., max=1.)

    def get_test_data(self, deg_num, save=False):
        """
        Writer : parkis

        Get degraded data and degradation label

        Args:
            deg_num (int) : the number of degradation in test data
            save (bool) : whether save test data and test label or not
        Returns:
            test_data (np.array) : data degraded from original test data which is not degraded
            test_label (np.array) : whether degradation exists or not
        """
        assert self.seed is not None, 'It is not a test dataset.'
        
        test_data = self.data.copy()
        test_label = np.zeros(len(self.data))

        for index in range(deg_num):
            first_index = np.random.choice(self.valid_index_list) # model input으로 들어갈 수 있는 index 중 하나 선택
            x, x_anomaly, _ = self.get_anomaly_data(index, first_index)
            test_data[first_index:first_index+self.data_seq_len] = x.numpy()
            test_label[first_index:first_index+self.data_seq_len] = x_anomaly.numpy()

        test_data = test_data.astype(np.float32)
        test_label = test_label.astype(np.int32)

        if save:
            print('Test data save!')
            np.save(os.path.join(self.options.save_folder, self.options.dataset + '_degraded.npy'), test_data)
            np.save(os.path.join(self.options.save_folder, self.options.dataset + '_degraded_label.npy'), test_label)

        return test_data, test_label