#!/bin/bash

# python cnu_classification.py --checkpoint /home/parkis/AnomalyDetection/anomalybert4ESS/AnomalyBERT/logs/230316204002_ESS_sionyu/state_dict.pt'



# ## from scratch ##
# for try in 1 2 3 4 5
# do
#     for lr in 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9
#     do
#         python cnu_classification.py --gpu_id=0 --lr=$lr --schedule_style=steplr --dataset=ESS_CNU_classification_scratch &
#         python cnu_classification.py --gpu_id=0 --lr=$lr --schedule_style=cosine --dataset=ESS_CNU_classification_scratch &
    
#         python cnu_classification.py --gpu_id=1 --lr=$lr --schedule_style=steplr --dataset=ESS_CNU_classification_scratch &
#         python cnu_classification.py --gpu_id=1 --lr=$lr --schedule_style=cosine --dataset=ESS_CNU_classification_scratch &

#         python cnu_classification.py --gpu_id=2 --lr=$lr --schedule_style=steplr --checkpoint '/home/parkis/AnomalyDetection/anomalybert4ESS/AnomalyBERT/logs/230316204002_ESS_sionyu/state_dict.pt' &
#         python cnu_classification.py --gpu_id=2 --lr=$lr --schedule_style=cosine --checkpoint '/home/parkis/AnomalyDetection/anomalybert4ESS/AnomalyBERT/logs/230316204002_ESS_sionyu/state_dict.pt'
#         wait
#     done
# done



# ## with pretrained ##
# for try in 1 2 3 4 5
# do
#     for lr in 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9
#     do
#         python cnu_classification.py --gpu_id=2 --lr=$lr --schedule_style=steplr --checkpoint '/home/parkis/AnomalyDetection/anomalybert4ESS/AnomalyBERT/logs/230316204002_ESS_sionyu/state_dict.pt'
#         python cnu_classification.py --gpu_id=2 --lr=$lr --schedule_style=cosine --checkpoint '/home/parkis/AnomalyDetection/anomalybert4ESS/AnomalyBERT/logs/230316204002_ESS_sionyu/state_dict.pt'
#     done
# done




### Test ###
python cnu_classification_test.py --checkpoint /data/ess/output/Anomaly_Detection/logs/ESS_CNU_classification_JJH_0/state_dict.pt