import os
import glob
from flask import Flask, jsonify, request
from deepant_pred import AnomalyDetector, DeepAnt, DataModule, TrafficDataset
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pandas import Timestamp
import pytorch_lightning as pl
import json
import torch.utils.data as data_utils
import joblib
import time
from usad_predict import USAD_pred

SEQ_LEN= 10
THRESHOLD = 0.87
DIR_PATH = 'deepant/'

model = DeepAnt(SEQ_LEN, 1) 
anomaly_detector = AnomalyDetector.load_from_checkpoint(DIR_PATH+'DeepAnt-best-checkpoint.ckpt', model = model)

USAD_SCALER_PATH = 'usad/usad_scaler.pkl'
USAD_MODEL_PATH = 'usad/model.pth'
USAD_THRESHOLD = 0.07
usad_model = USAD_pred(USAD_SCALER_PATH, USAD_MODEL_PATH)

app = Flask(__name__)

def DataLoader_deepAnt(data):
    '''
    DeepAnT - DataLoader
     * The code below is adapted from a GitHub repository.
     * URL: https://github.com/datacubeR/DeepAnt
     * Feel free to use and modify this code, but make sure to check the original sources for any updates or improvements.
    '''
    df = pd.DataFrame(list(data.items()), columns=['timestamp', 'value']).set_index('timestamp')
    dataset = TrafficDataset(df, SEQ_LEN)
    target_idx = dataset.timestamp 
    dm = DataModule(df, SEQ_LEN)
    return dataset, target_idx, dm


def usad_predict(data):
    '''
    Audibert, J., Michiardi, P., Guyard, F., Marti, S., Zuluaga, M. A. (2020).
    USAD : UnSupervised Anomaly Detection on multivariate time series.
    Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, August 23-27, 2020
    
    '''
    data = np.array(list(data.values())).reshape(-1,1)
    test_loader = usad_model.load_dataset(data)
    result = usad_model.anomaly_detection(test_loader)
    
    return result




@app.route('/predict_deepant', methods=['GET','POST'])
def predict():
    data = request.json
    trainer = pl.Trainer()
    dataset_, target_idx_, dm_ = DataLoader_deepAnt(data)
    output = trainer.predict(anomaly_detector, dm_)
    preds_losses = pd.Series(torch.tensor([item[1] for item in output]).numpy(), index = target_idx_)
    anomaly_score = preds_losses.loc[lambda x: x > THRESHOLD].to_dict()
    return anomaly_score



@app.route('/predict_usad', methods=['GET','POST'])
def usad():
    data = request.json


    result = usad_predict(data)
    anomaly_score = True if result > USAD_THRESHOLD else False
    
    
    anomaly = {}
    anomaly[list(data.keys())[0]] = anomaly_score

    
    return json.dumps(anomaly)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2599)  


