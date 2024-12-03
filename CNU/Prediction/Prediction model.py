import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dropout,Dense,Activation
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam


# 데이터 파일 로드
hi_soh_data = pd.read_csv('경로 설정')
hi_soh_data = hi_soh_data.to_numpy()

print(hi_soh_data)
hi_soh_data.shape

#변수 생성
hi_data = hi_soh_data[:,0:300]
soh = hi_soh_data[:,-1]

#나누기
x_train = hi_data[:979, 1:] 
y_train = soh[:979]         

x_test = hi_data[979:, 1:]  
y_test = soh[979:]  

plt.plot(x_train)
