CUDA_VISIBLE_DEVICES=3 python indicator.py \
                        --seed=0 \
                        --base_seconds=20   --base_model=res18 --base_decay=0.95   \
                        --phase1_seconds=20 --phase1_model=res10 --phase1_decay=0.95 \
                        --phase2_seconds=20 --phase2_model=res10 --phase2_decay=0.95
