
SECONDS="200 100 50 45 40 35 30 25 20 15 10"
LOSS_FN="L1Loss"
MODEL="cnn res10 res18 res34"
EPOCHS="200"
SEED="0 1 2 3 4 5 6 7 8 9"
DECAY="0.98 0.95 0.9 0.8"


for seconds in $SECONDS
do
    for loss_fn in $LOSS_FN
    do
        for model in $MODEL
        do
            for epochs in $EPOCHS
            do
                for seed in $SEED
                do
                    CUDA_VISIBLE_DEVICES=0 python main.py --config=config_VITdV_OCV.yaml --seconds=$seconds --loss_fn=$loss_fn --model=$model --epochs=$epochs  --seed=$seed --decay=0.98 &
                    CUDA_VISIBLE_DEVICES=1 python main.py --config=config_VITdV_OCV.yaml --seconds=$seconds --loss_fn=$loss_fn --model=$model --epochs=$epochs  --seed=$seed --decay=0.95 &
                    CUDA_VISIBLE_DEVICES=2 python main.py --config=config_VITdV_OCV.yaml --seconds=$seconds --loss_fn=$loss_fn --model=$model --epochs=$epochs  --seed=$seed --decay=0.9  &
                    # CUDA_VISIBLE_DEVICES=3 python main.py --config=config_VITdV_OCV.yaml --seconds=$seconds --loss_fn=$loss_fn --model=$model --epochs=$epochs  --seed=$seed --decay=0.8  &
                done
                wait
            done
        done
    done
done