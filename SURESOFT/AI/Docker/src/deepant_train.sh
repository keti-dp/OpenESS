python main.py --dataset "./data/train.csv" --mode "train" --model "deepant" --lr 1e-3 --n_epochs 150 --batch_size 2048 --features  "CPU usage" "Heap current" "Time stamp" --es_epochs 10



python main.py --dataset "./data/test.csv" --mode "pred" --model "deepant" --lr 1e-3 --n_epochs 150 --batch_size 2048 --features "CPU usage" "Heap current" "Time stamp" --es_epochs 15
