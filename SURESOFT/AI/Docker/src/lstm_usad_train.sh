#Train
python main.py --dataset './data/train.csv' --mode "train" --model "lstm_usad" --lr 1e-3 --n_epochs 300 --batch_size 2048 --features "CPU usage" --es_epochs 300


#Predict
python main.py --dataset "./data/test.csv" --mode "pred" --model "lstm_usad" --lr 1e-4 --n_epochs 150 --batch_size 1 --features "CPU usage" --es_epochs 15
