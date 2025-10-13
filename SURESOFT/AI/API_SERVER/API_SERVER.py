from typing import Union, Optional, List
import sys
import os
import glob
import json


import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from USAD import USAD_train, USAD_pred
from DeepAnt import DeepAnt_pred, DeepAnt_train



class USADTrainSchema(BaseModel):
    dataset_path: str = "Train Dataset"
    dim: int = 128
    hidden_size: int = 300
    lr: float = 1e-5
    n_epochs: int = 150
    batch_size: int = 2048
    features: Union[str, List[str]] = "CPU usage"
    es_epochs: int = 15


class USADPredSchema(BaseModel):

    dataset_path: str = "Test Dataset"
    threshold: Optional[float] = "null"



class DeepAntTrainSchema(BaseModel):
    dataset_path: str = "Train Dataset"
    lr: float = 1e-3
    n_epochs: int = 150
    batch_size: int = 2048
    features: Union[str, List[str]] = "CPU usage"
    es_epochs: int = 10


class DeepAntPredSchema(BaseModel):
    dataset_path: str = "Test Dataset"
    threshold: Optional[float] = None
    
    
    
app = FastAPI()


sys.stdout = open(os.devnull, 'w')
@app.get("/")
async def index():
    
    return {"message":"BMS Anomaly Detection Model API PAGE"}


@app.post(
    "/usad/train/",
    responses={
        404: {"description": "Not Found Exception"},
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def USAD_train_call(param: USADTrainSchema = USADTrainSchema):
    
    id_num = len(glob.glob('./model/usad_model*.pth'))+1
    try:
        DATASET_PATH = str(param.dataset_path)
        DIM = int(param.dim)
        HIDDEN_SIZE = int(param.hidden_size)
        LR = float(param.lr)
        N_EPOCHS = int(param.n_epochs)
        BATCH_SIZE = int(param.batch_size)
        FEATURES = param.features
        ES_EPOCHS = int(param.es_epochs)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Type Error")
    try:
        data, FEATURES = read_data(DATASET_PATH, FEATURES)
    except Exception:
        raise HTTPException(status_code=404, detail="NOT Found Exception")
    usad = USAD_train(
        batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, lr=LR, hidden_size=HIDDEN_SIZE
    )
    try:
        data = usad.fit_scaler(data, id_num)
        train_loader, val_loader = usad.load_dataset(data, DIM, FEATURES)
        usad.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            lr=LR,
            es_epochs=ES_EPOCHS,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")
    usad.save_model(id_num)
    return_result = {"id":id_num, "result":"USAD Model Training Done"}
    return return_result


@app.get(
    "/usad/pred/{model_id}",
    responses={
        404: {"description": "Not Found Exception"},
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def USAD_pred_call(model_id:int, dataset:str = './data/test.csv'):
    try:
        DATASET_PATH = str(dataset)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Type Error")
    

    USAD_SCALER_PATH = f"./model/usad_scaler_{model_id}.pkl"
    USAD_MODEL_PATH = f"./model/usad_model_{model_id}.pth"

    # # data = data.iloc[128:128+1024]
    # # print(data)

    usad_model = USAD_pred(USAD_SCALER_PATH, USAD_MODEL_PATH)
    try:
        data, features = read_data(DATASET_PATH, usad_model.features)
    except Exception:
        raise HTTPException(status_code=404, detail="NOT Found Exception")
    try:
        test_loader, time_index = usad_model.load_dataset(data)
        results = usad_model.anomaly_detection(test_loader)

        time_index = list(np.datetime_as_string(time_index).flatten())
        result = {x: y for x, y in zip(time_index, results)}

        with open("./logs/USAD_Prediction.txt", "w") as f:
            f.write(json.dumps({x: y for x, y in zip(time_index, results)}, indent=4))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")
    return_result = {"id":model_id, "result":result}
    return return_result


@app.post(
    "/deepant/train/",
    responses={
        404: {"description": "Not Found Exception"},
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def DeepAnt_Train(param: DeepAntTrainSchema = DeepAntTrainSchema):
    id_num = len(glob.glob('./model/deepant_model*.pth')) + 1
    try:
        DATASET_PATH = str(param.dataset_path)
        LR = float(param.lr)
        N_EPOCHS = int(param.n_epochs)
        BATCH_SIZE = int(param.batch_size)
        FEATURES = param.features
        ES_EPOCHS = int(param.es_epochs)
    except Exception:
        HTTPException(status_code=400, detail="Invalid Type Value")

    try:
        data, FEATURES = read_data(DATASET_PATH, FEATURES)
    except Exception:
        raise HTTPException(status_code=404, deatil="Not Found Exception")
    try:
        deepant = DeepAnt_train(BATCH_SIZE, N_EPOCHS, LR, FEATURES)
        data = deepant.fit_scaler(data, id_num)
        train_loader, val_loader = deepant.load_dataset(data)
        deepant.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            lr=LR,
            es_epochs=ES_EPOCHS,
        )
        deepant.save_model(id_num)
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")
    return_result = {"model_id":id_num, "result":"deepant Model Training Done"}
    return return_result


@app.get(
    "/deepant/pred/{model_id}",
    responses={
        404: {"description": "Not Found Exception"},
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def DeepAnt_Predict(model_id:int = 1, dataset:str = './data/test.csv'):
    try:
        model_id = int(model_id)
        DATASET_PATH = str(dataset)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Type Value")
    

    DEEPANT_SCALER_PATH = f"./model/deepant_scaler_{model_id}.pkl"
    DEEPANT_MODEL_PATH = f"./model/deepant_model_{model_id}.pth"

    deepant_model = DeepAnt_pred(DEEPANT_SCALER_PATH, DEEPANT_MODEL_PATH)
    try:
        data, __ = read_data(DATASET_PATH, deepant_model.features)
    except Exception:
        HTTPException(status_code=404, deatil="Not Found Exception")
    try:
        test_loader, time_index = deepant_model.load_dataset(data)
        results = deepant_model.anomaly_detection(test_loader, 0.0)
        time_index = list(np.datetime_as_string(time_index).flatten())

        results = {x: y for x, y in zip(time_index, results)}

        with open("./logs/DeepAnt_Prediction.txt", "w") as f:
            f.write(json.dumps(results, indent=4))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")
    return_result = {"model_id":model_id, "result":results}
    return return_result


def read_data(path, features):
    
    data = pd.read_csv(path)
    if type(features) is list:
        features = features
    else:
        features = [features]
    try:
        data.Time = pd.to_datetime(data.Time)
    except Exception:
        data.Time = pd.to_datetime(data.Time.str.slice(1, -1))
    data = data.set_index(data.Time)
    data = data[features]
    return data, features
