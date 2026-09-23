#41-1 copy

import numpy as np
import pandas as pd
import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
(x_train, y_train), (x_test, y_test)= mnist.load_data()
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

# print(np.max(x_train), np.min(x_train)) # 255 0
# print(np.max(x_test), np.min(x_test)) # 255 0

#### 스케일링 1
x_train = x_train/255. # .만 붙이면 float 형태로 출력하게 됨
x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0 (0~1 사이로 나옴)
# print(np.max(x_test), np.min(x_test)) # 1.0 0.0


# ##### X reshape  하는 이유 : input_shape 
# x_train = x_train.reshape(-1,28,28,1) 
# x_test = x_test.reshape(-1,28,28,1)
# print(x_train.shape ,x_test.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)

x_train = x_train.reshape(-1,28*28*1) 
x_test = x_test.reshape(-1,28*28*1)
# print(x_train.shape ,x_test.shape)  # (60000, 784) (10000, 784)

# exit()

##### y값 알아보기

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))

##### 원핫인코더(분류)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False) 

y_train = y_train.reshape(-1,1) # 데이터갯수를 안다면 y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_train.shape, y_test.shape) #(60000, 10) (10000, 10)

# 실습시작 : CNN 보다 성능 높이게


#2. 모델구성
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(10, activation='softmax'))  

model.summary()
# exit()

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
# learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss="categorical_crossentropy", 
              optimizer=Adam(learning_rate=learning_rate),
              metrics = ['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    verbose=1,
    patience=20,
)

import datetime
date = datetime.datetime.now()
date = date.strftime('%d%m-%H%M')

path ='./_save/keras36/'

filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k36_", date, "-", filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    save_best_only=True,
    verbose=1,
    filepath=filepath,
    mode='min',
)

start_time=time.time()
model.fit(x_train,y_train,
          epochs=500, 
          batch_size=32, 
          verbose=1,
          validation_split=0.2,
          callbacks = [es, mcp],
          )
end_time=time.time()

# 4. 평가, 예측
print( "=============model.evaluate=================")
loss = model.evaluate(x_test, y_test, verbose=1)

print('loss : ', round(loss[0],2))
print('acc : ', round(loss[1],2))

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print('acc_score : ', acc_score)
print ('걸린시간 : ', round(end_time-start_time,2),'초' )

### CNN 최고성능
# loss :  0.02
# acc :  0.99
# acc_score :  0.9942
# 걸린시간 :  180.95 초

### DNN 3차 (GPU)
# loss :  0.1
# acc :  0.98
# acc_score :  0.9761
# 걸린시간 :  91.08 초


# learning_rate = 0.005
# Epoch 27: early stopping
# loss :  0.17
# acc :  0.96
# acc_score :  0.9608
# 걸린시간 :  83.56 초



















