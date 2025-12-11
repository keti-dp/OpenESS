# AnomalyBERT: Transformer-based Anomaly Detector

This is the code for **Self-supervised Transformer for Time Series Anomaly Detection using Data Degradation Scheme**.
Especially, code for **AnomalyBERT4ESS** is a code for *<U>Development of intelligent SW framework for safe autonomous operation and performance evaluation of large-scale distributed energy storage infrastructure</U>*.

## Installation

Please clone our repository at `path/to/repository/` and install the packages in `requirements.txt`.
Before installing the packages, we recommend installing Python 3.8 and Pytorch 1.9 with CUDA.

```
git clone https://github.com/Jhryu30/AnomalyBERT4ESS.git path/to/repository/

conda create --name your_env_name python=3.8
conda activate your_env_name

pip install torch==1.9.0+cuXXX -f https://download.pytorch.org/whl/torch_stable.html  # cuXXX for your CUDA setting
pip install -r requirements.txt
```

We use five public datasets, SMAP, MSL, SMD, SWaT, and WADI.
Following the instruction in [here](utils/DATA_PREPARATION.md), you can download and preprocess the datasets.
After preprocessing, you need to edit your dataset directory in `utils/config.py`.

```
DATASET_DIR = 'path/to/dataset/processed/'
```

## Training

We provide the training code for our model.
(recommended) For example, to train a model of 6-layer Transformer body on ESS_sionyu dataset, run:

```
python3 train.py --dataset=ESS_sionyu
```

To train a model on ESS_panli dataset with patch size of 2 and customized outlier synthesis probability, run:

```
python3 train.py --dataset=ESS_panli --patch_size=2 --soft_replacing=0.5 --uniform_replacing=0.1 --peak_noising=0.1
```


If you want to customize the model and training settings, please check the options in `train.py`.

## Anomaly score estimation and metric computation

To estimate anomaly scores and compute ess-score with and without the point adjustment of test data with the trained model, run the `estimate.py` code.
For example, you can estimate anomaly scores of ESS_sionyu test set.

```
python3 estimate.py --dataset=ESS_sionyu --base_folder folder directory where the logs and weights of the trained model are stored
```

If you want to customize the estimation or computation settings, please check the options in `estimate.py`.
