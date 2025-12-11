import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import interp1d
import ray

PATH = '/data/ev/NASA/processed/'


@ray.remote
def select_dis_visoh(file_num: int) -> None:

    # (0) Load parquet
    df = pd.read_parquet(os.path.join(PATH, f'RW{file_num:02d}.parquet'))

    # (1) Random Walk Discharge -> V, I, SOH (1 sec interval)
    _dis = df.loc[(df['charge_type'] == 'discharge (random walk)') | (df['charge_type'] == 'rest (random walk)')]
    _start = _dis.loc[_dis['N'] - _dis['N'].shift(1) > 1].index.to_list()
    _end = _dis.loc[_dis['N'].shift(-1) - _dis['N'] > 1].index.to_list()
    _start.insert(0, _dis.index[0])
    _end.append(_dis.index[-1])

    data = []
    for s, e in zip(_start, _end):
        _df = df.loc[s:e].copy()
        if len(_df) < 128:
            continue
        vi = _df[['V', 'I']].values
        soh = _df['SOH'].iloc[-1]
        data.append((soh, vi))
        
    np.savez(
        f'./encoder_data/RW{file_num:02d}_rw_dis_V_I_SOH.npz',
        data=np.array(data, dtype=object),
    )

    # (2) Reference type B Discharge -> V, I, SOH (10 sec -> 1 sec interval)
    _dis = df.loc[(df['charge_type'] == 'reference discharge') & (df['I'] < -0.5)]
    _start = _dis.loc[_dis['N'] != _dis['N'].shift(1)].index.to_list()
    _end = _dis.loc[_dis['N'] != _dis['N'].shift(-1)].index.to_list()

    data = []
    for s, e in zip(_start, _end):
        _df = df.loc[s:e].copy()
        _df['t'] -= _df['t'].iloc[0]
        _df['t'] = _df['t'].apply(lambda x: round(x))
        t = _df['t'].values
        f_v = interp1d(t, _df['V'].values)
        f_i = interp1d(t, _df['I'].values)
        new_t = np.arange(_df['t'].iloc[-1] + 1)
        new_v = f_v(new_t)
        new_i = f_i(new_t)
        vi = np.stack([new_v, new_i], axis=1)
        soh = _df['SOH'].iloc[-1]
        data.append((soh, vi))

    np.savez(
        f'./encoder_data/RW{file_num:02d}_ref_dis_V_I_SOH.npz',
        data=np.array(data, dtype=object),
    )
    return


if __name__ == '__main__':

    ray.init(num_cpus=40)
    os.makedirs('./encoder_data/', exist_ok=True)

    start = datetime.now()
    tasks = []
    for i in range(1, 29):
        tasks.append(select_dis_visoh.remote(i))
    for task in tasks:
        ray.get(task)
    print(f'Elapsed: {datetime.now() - start}')

    ray.shutdown()