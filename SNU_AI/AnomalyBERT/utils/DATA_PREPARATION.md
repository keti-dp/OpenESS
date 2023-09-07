# Data Preparation for AnomalyBERT

This is an instruction for preprocessing of ESS; sionyu, panli_bank1, panli_bank2.
Before you download dataset files, please set your dataset folder `path/to/dataset/` and then follow the instructions below.

## Data Download

The structure of your dataset directory should be as follows:

```
path/to/dataset/
|-- ESS_sionyu
|-- ESS_panli_bank1
|-- ESS_panli_bank2
```

## Data Preprocessing

If you finish downloading the entire (or a part of) datasets, you need to preprocess them.

After preprocessing, you will have three npy files `{dataset}_train.npy`, `{dataset}_test.npy`, and `{dataset}_test_label.npy` in the `path/to/dataset/processed/` folder.
Please write down the path to you processed dataset directory in `utils/config.py` as below.

```
DATASET_DIR = 'path/to/dataset/processed/'
```