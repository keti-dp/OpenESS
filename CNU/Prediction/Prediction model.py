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

