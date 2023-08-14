import requests
import json
import pandas as pd
import websockets
import asyncio
import time
from multiprocessing import Process, Pipe, freeze_support,Queue
import os


def preprocess_data(df):    
    '''
    Code Change 필요 (Input Data 형식 따라)
    '''
    df = df.rename(columns= {'Time':'timestamp',
                             'CPU usage':'CPU',
                            })
    df = df[df.timestamp != 'time']
    df = df.astype({'CPU':float})
    df.timestamp = pd.to_datetime(df.timestamp.apply(lambda x: str(x)[1:-1]))
    
    df = df.set_index('timestamp')
    return df

def send_api(path, method, recv_pipe):
    '''
    key: timestamp
    value: CPU                 --> 나중에 Memory 추가
    type: isoformat(timestamp): float(CPU)
    '''
    API_HOST = "http://10.10.10.61:2599"
    url = API_HOST + path
    headers = {'Content-Type': 'application/json', 'charset': 'UTF-8', 'Accept': '*/*'}
    
    
    while True:
        data_ = recv_pipe.recv()
        
        
        
        
        body = {k.isoformat():v for k,v in data_.CPU.to_dict().items()}
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, headers=headers, data=json.dumps(body, ensure_ascii=False, indent="\t"))
                   
            recv_pipe.send(response.text)
        
        except Exception as ex:
            print(ex)




    
        
  
  
class WebSocketServer:
    def __init__(self, recv_pipe):
        self.recv_pipe = recv_pipe 
        start_server = websockets.serve(self.handle_data, '10.10.10.61', 1223)    
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
        
        
    async def handle_data(self,websocket):
        while True:
            
            data = self.recv_pipe.recv()
            os.system('clear')
            print(data)
            await websocket.send(data)
            await asyncio.sleep(0)




if __name__=='__main__':
    freeze_support()

    recv_pipe, send_pipe = Pipe()
    deep_ant_send, deep_ant_recv = Pipe()
    usad_send, usad_recv = Pipe()
    
    
    Process(target = WebSocketServer, args = (recv_pipe,)).start()
    Process(target = send_api, args = ("/predict_deepant", "POST", deep_ant_recv,)).start()
    Process(target = send_api, args = ("/predict_usad", "POST", usad_recv,)).start()
    
    DIR_PATH = '/STORAGE/ESS/01_DATA/TESTBED/0725_data_sc2_v1_통합.csv'
    DIR_PATH = 'data/test_data.csv'
    data = pd.read_csv(DIR_PATH)
    data = preprocess_data(data)
    result = {}
    
    i = 1
    while True:
        df = data.iloc[[i]]
        
        result['data'] = df.to_dict(orient = 'list')
        result['data']['timestamp'] = [df.index[0].isoformat()]
        result['DeepAnt'] = None
        result['USAD'] = None
        result['reset'] = False
        
        
        if i % 20 == 0:
            deep_ant_send.send(data.iloc[i-20:i])
            
        if i % 128 == 0:
            usad_send.send(data.iloc[i-128:i])
            
        
        
        if deep_ant_send.poll():
            result['DeepAnt'] = json.loads(deep_ant_send.recv())
        
        if usad_send.poll():
            result['USAD'] = json.loads(usad_send.recv())
        
        
        
        
        i += 1    
        if i == len(data):
            i = 1
            result['reset'] = True
        send_pipe.send(json.dumps(result, indent= 4))
        time.sleep(0.1)
        
        
        
        
            
        


