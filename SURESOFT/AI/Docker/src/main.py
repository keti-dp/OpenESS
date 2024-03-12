#-*- coding: utf-8 -*-
import argparse
import json
import os

import pandas as pd
import numpy as np

from DeepAnt import DeepAnt_train, DeepAnt_pred
from USAD import USAD_train, USAD_pred








def main(options):
    
    
    
    
    # Define Dataset
    DATA_L = None
    if str(options.dataset).endswith('/'):
        DATA_L = options.dataset + '*'
    elif str(options.dataset).endswith('*'):
        DATA_L = options.dataset
    elif str(options.dataset).endswith('.csv'):
        DATA = options.dataset
    else:
        DATA_L = options.dataset + '/*'
        
    # Set Window_Size
    if options.model == 'usad':
        DIM = 128
    else:
        DIM = 10
        
    # Set Features
    if type(options.features) == list:
        FEATURES = options.features        
    else:
        FEATURES = [options.features]
        
    BATCH_SIZE = options.batch_size
    N_EPOCHS = options.n_epochs
    LR = options.lr
    N_FEATURES = len(FEATURES)
    ES_EPOCHS = options.es_epochs
    
    
    
    
    
    
    
    if DATA_L== None:
        data = pd.read_csv(DATA)
        data = data[data['CPU usage'] > 0]
        data.Time = pd.to_datetime(data.Time)
        data = data.set_index(data.Time)
        data = data[FEATURES]
        print(data.shape)
    else:
        raise NotImplementedError
            
   
    
    
    
    if options.model == "usad":
        
        if options.mode == 'train': 
            #Train Param
            
            
            usad = USAD_train(BATCH_SIZE, N_EPOCHS,LR)
            
            data = usad.fit_scaler(data)
            train_loader, val_loader = usad.load_dataset(data, DIM, FEATURES)
            history = usad.fit(train_loader=train_loader, val_loader=val_loader, lr = LR, es_epochs = ES_EPOCHS)
            usad.save_model()
        
        elif options.mode == "pred":
            
            USAD_SCALER_PATH ='./model/usad_scaler.pkl'
            USAD_MODEL_PATH = './model/usad_model.pth'
            USAD_THRESHOLD = options.threshold
            
            
            # data = data.iloc[128:128+1024]
            # print(data)
            
            usad_model = USAD_pred(USAD_SCALER_PATH, USAD_MODEL_PATH)

            test_loader, time_index = usad_model.load_dataset(data)
            results = usad_model.anomaly_detection(test_loader)        
            
            time_index = list(np.datetime_as_string(time_index).flatten())
            
            
            result = (json.dumps({x:y for x,y in zip(time_index, results)}, indent = 4))
            with open('../logs/USAD_Prediction.txt','w') as f:
                f.write(result)
            
            
            return result
            
                        
            
    elif options.model == "deepant" :
                
        if options.mode == "train" :
            
            # train 
            
            
            deepant = DeepAnt_train(BATCH_SIZE, N_EPOCHS, LR, N_FEATURES)
            data = deepant.fit_scaler(data)
            train_loader, val_loader = deepant.load_dataset(data)
            history = deepant.fit(train_loader=train_loader, val_loader=val_loader,lr = LR, es_epochs = ES_EPOCHS)
            deepant.save_model()
        
            
        elif options.mode == "pred" :
            
            
            DEEPANT_SCALER_PATH = './model/deepant_scaler.pkl'
            DEEPANT_MODEL_PATH = './model/deepant_model.pth'
            DEEPANT_THRESHOLD = options.threshold
            
                       
            deepant_model = DeepAnt_pred(DEEPANT_SCALER_PATH, DEEPANT_MODEL_PATH)
            test_loader, time_index = deepant_model.load_dataset(data)
            results = deepant_model.anomaly_detection(test_loader)
            time_index = list(np.datetime_as_string(time_index).flatten())
            
              
            
            result = (json.dumps({x:y for x,y in zip(time_index, results)}, indent= 4))
            with open('../logs/DeepAnt_Prediction.txt','w') as f:
                f.write(result)
                
            
            return result

    
    
    
    





if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    
    # Dataset Path
    parser.add_argument("--dataset", default='./data/train.csv', type=str, help='Path to Dataset') 
    parser.add_argument("--mode", default = "train", type=str, help='Choose train or predict', choices = ['train','pred'])
    
    # Select Model
    parser.add_argument("--model", default = "usad", type = str, help= "Choose usad or deepAnt", choices=['usad','deepant'], required=True)
    
    # Train Parameters
    parser.add_argument("--lr", default=1e-5,help = 'Learning Rate', type=float)
    parser.add_argument("--n_epochs", default=150, type=int, help='Training_steps')
    parser.add_argument("--batch_size", default=1280, type=int, help='Set Batch_Size')
    parser.add_argument("--features", default='CPU usage', nargs='+',type=str, help='Columns for Model')
    parser.add_argument("--es_epochs", default=10,type=int, help='Early Stopping Epochs')
    
    

    # Predict Parameter
    parser.add_argument("--threshold", default = 0.04, type=float, help='Threshold')
    
    
    options = parser.parse_args()
            
    main(options)
            
            
