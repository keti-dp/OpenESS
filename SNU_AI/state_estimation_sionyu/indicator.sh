
SECONDS="15 20 25 30 35 40 45 50"
LOSS_FN="L1Loss"
MODEL="cnn res10 res18"
SEED="0 1 2 3 4 5 6 7 8 9"
DECAY="0.98 0.95 0.9 0.8"


# for seconds in $SECONDS
# do
#     for model1 in $MODEL
#     do
#         for model2 in $MODEL
#         do
#             for model3 in $MODEL
#             do
#                 for seed in $SEED
#                 do
#                     CUDA_VISIBLE_DEVICES=1 python indicator.py \
#                         --seed=$seed \
#                         --base_seconds=$seconds   --base_model=$model1 --base_decay=0.95   \
#                         --phase1_seconds=$seconds --phase1_model=$model2 --phase1_decay=0.95 \
#                         --phase2_seconds=$seconds --phase2_model=$model3 --phase2_decay=0.95 &
#                     sleep 1
                    
#                     CUDA_VISIBLE_DEVICES=2 python indicator.py \
#                         --seed=$seed \
#                         --base_seconds=$seconds   --base_model=$model1 --base_decay=0.9   \
#                         --phase1_seconds=$seconds --phase1_model=$model2 --phase1_decay=0.9 \
#                         --phase2_seconds=$seconds --phase2_model=$model3 --phase2_decay=0.9 &
#                     sleep 1

#                     CUDA_VISIBLE_DEVICES=3 python indicator.py \
#                         --seed=$seed \
#                         --base_seconds=$seconds   --base_model=$model1 --base_decay=0.8   \
#                         --phase1_seconds=$seconds --phase1_model=$model2 --phase1_decay=0.8 \
#                         --phase2_seconds=$seconds --phase2_model=$model3 --phase2_decay=0.8 &
#                     sleep 1

#                 done
#                 wait
#             done
#         done
#     done
# done

## gpu 2개 버전
for seconds in $SECONDS
do
    for model1 in $MODEL
    do
        for model2 in $MODEL
        do
            for model3 in $MODEL
            do
                for seed in $SEED
                do
                    CUDA_VISIBLE_DEVICES=0 python indicator.py \
                        --seed=$seed \
                        --base_seconds=$seconds   --base_model=$model1 --base_decay=0.95   \
                        --phase1_seconds=$seconds --phase1_model=$model2 --phase1_decay=0.95 \
                        --phase2_seconds=$seconds --phase2_model=$model3 --phase2_decay=0.95 &
                    sleep 1
                    
                    CUDA_VISIBLE_DEVICES=3 python indicator.py \
                        --seed=$seed \
                        --base_seconds=$seconds   --base_model=$model1 --base_decay=0.9   \
                        --phase1_seconds=$seconds --phase1_model=$model2 --phase1_decay=0.9 \
                        --phase2_seconds=$seconds --phase2_model=$model3 --phase2_decay=0.9 &
                    sleep 1
                done
                wait
            done
        done
    done
done


for seconds in $SECONDS
do
    for model1 in $MODEL
    do
        for model2 in $MODEL
        do
            for model3 in $MODEL
            do
                for seed in $SEED
                do
                    CUDA_VISIBLE_DEVICES=3 python indicator.py \
                        --seed=$seed \
                        --base_seconds=$seconds   --base_model=$model1 --base_decay=0.8   \
                        --phase1_seconds=$seconds --phase1_model=$model2 --phase1_decay=0.8 \
                        --phase2_seconds=$seconds --phase2_model=$model3 --phase2_decay=0.8 &
                    sleep 1
                done
                wait
            done
        done
    done
done