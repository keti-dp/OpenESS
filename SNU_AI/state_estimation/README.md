# ESS SOH Estimation
This is a resource for estimation on SOH from ESS data.
## Usage
Please install the packages in `requirements.txt` first.

```
pip3 install -r requirements.txt
```

Before running, you need to write your own directory that contains processed nasa datasets to `DATA_DIR` in `utils/config.py`.
To run the training process, execute `train.py` file with `python3` operation.
Here are some examples for training RNN. Note that you should write the model name to train explicitly.

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
python3 train.py --model=nasa_rnn --test_model="directory/of/trained/model/log/" --log_test_history
```


To see another options, use `--help` command or check `utils/options.py` file please.