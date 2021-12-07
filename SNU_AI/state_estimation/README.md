# ESS SOH Estimation
This is a resource for estimation on SOH from ESS data.
## Usage
### Training on NASA datasets
Please install the packages in `requirements.txt` first.

```
pip3 install -r requirements.txt
```

Before running, you need to write down your own directory that contains raw nasa datasets and processed nasa datasets.
In `utils/config.py`, please edit `RAW_DATA_DIR` to the directory containing raw nasa datasets (`RW01.parquet - RW28.parquet`), and `PROCESSED_DATA_DIR` to the directory that processed nasa datasets will be located in.
It is recommanded to set `PROCESSED_DATA_DIR` to be an empty folder.

To make processed nasa datasets, enter the line below on a terminal.
```
python3 nasa_preprocessing.py
```
The processed nasa datasets will be located in `PROCESSED_DATA_DIR`.

To run the training process, execute `train.py` file with `python3` operation.
Here are some examples for training RNN. Note that you should write the name of model to train explicitly.

Training RNN
```
python3 train.py --model=nasa_rnn
```
Training LSTM with learning rate 0.0001, sliding window 128, number of training epochs 300, using data columns `V`, `I`, `T`, `dt`.
```
python3 train.py --model=nasa_lstm --lr=0.0001 --sliding_window=128 --num_epochs=300 --columns="V,I,T,dt"
```

Trained model and its information are logged in `logs/` directory. You can also test this trained model with the log file.
```
python3 train.py --model=nasa_rnn --test_model="directory/of/trained/model/log/"
```


To see another options, use `--help` command or check `utils/options.py` file please.