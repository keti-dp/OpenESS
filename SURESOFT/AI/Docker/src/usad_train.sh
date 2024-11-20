python main.py --dataset "./data/train.csv" --mode "train" --model "usad" --lr 1e-4 --n_epochs 150 --batch_size 1024 --features "CPU usage" "Heap current" "Time stamp" --es_epochs 15


python main.py --dataset "./data/test.csv" --mode "pred" --model "usad" --lr 1e-4 --n_epochs 150 --batch_size 1024 --features "CPU usage" "Heap current" "Time stamp"    --es_epochs 15 #
