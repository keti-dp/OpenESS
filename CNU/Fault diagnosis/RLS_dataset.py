## 라이브러리
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pylab as plt 
import scipy.io
from scipy import stats
from scipy import io
from scipy.io import loadmat
from matplotlib.ticker import ScalarFormatter
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers



# 배터리 정상/과충전/과방전 모듈(6S2P) 전압, 전류 데이터 셋 기반 내부 모델 파라미터(Ri, Rdiff, Cdiff) 추출

# Load OCV data
OCV = '경로 설정 필요'
# 셀 level OCV는 OCV.csv 파일 사용

SOC = np.linspace(1, 0, num=21)

normal_dir = '경로 설정 필요'
overcharge_dir = '경로 설정 필요'
overdischarge_dir = '경로 설정 필요'
save_dir = '경로 설정 필요'

# 정상, 과충전 및 과방전 1~100사이클 충전 및 방전 구분해서 코드 실행 필요
# ex) 첨부된 정상, 과충전, 과방전 팩 100사이클 충전/방전으로 dir 설정해서 코드 실행
# 셀은 1~6열(셀 전압), 7열(모듈 전압), 8열(모듈 전류)
