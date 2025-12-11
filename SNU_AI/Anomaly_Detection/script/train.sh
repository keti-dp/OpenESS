"""
Anomaly Detection
작업 위치 : /home/ess/year345/Anomaly_Detection
데이터 관련 config 위치 : ./utils/config.py



"""
# 훈련
python train.py --gpu_id 1

# 전이학습
# finetune(시온유 -> 판리)
python train_lora.py --gpu_id 1 --dataset ESS_panlil --checkpoint logs/.../state_dict.pt

# Anomaly Score 및 (NiN+FiF)/2 계산(state_dict의 dataset으로 test)
python estimate.py --gpu_id 1
