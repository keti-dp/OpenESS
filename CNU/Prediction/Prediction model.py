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

# NaN 대체
for i in range(1, x_train.shape[0]): 
    mask = np.isnan(x_train[i]) 
    x_train[i, mask] = x_train[i - 1, mask]

print(np.isnan(x_train).sum())

# 데이터 Reshape
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# 모델 생성
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape = x_train.shape[1:]))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

learning_rate = 0.0001 
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))


# SOH 추정
soh_predictions = model.predict(x_test)

print(soh_predictions[0])
print(y_test)

# SOH 예측 결과 출력
for i in range(len(soh_predictions)):
    print(f"실제 SOH: {y_test[i]:.4f}, 예측 SOH: {soh_predictions[i][0]:.4f}")

# 학습 손실 및 예측 결과 Plot
plt.figure(figsize=(20, 7))

# 학습 손실
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 학습 손실 및 예측 결과 Plot
plt.figure(figsize=(20, 7))


# SOH 예측 비교
plt.subplot(1, 2, 2)
plt.plot(y_test, label='Actual SOH')
plt.plot(soh_predictions, label='Predicted SOH')
plt.title('Actual vs Predicted SOH')
plt.xlabel('Cycle')
plt.ylabel('SOH')
plt.legend()

plt.tight_layout()
plt.show()

