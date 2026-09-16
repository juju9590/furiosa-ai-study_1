
# 실습 acc : 0.67 (n, 32,32,3)

from tensorflow.keras.datasets import cifar10

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
(x_train, y_train), (x_test, y_test)= cifar10.load_data()
# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

# print(np.max(x_train), np.min(x_train)) # 255 0
# print(np.max(x_test), np.min(x_test)) # 255 0

#### 스케일링 1
x_train = x_train/255. 
x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0 
# print(np.max(x_test), np.min(x_test)) # 1.0 0.0

##### y값 알아보기
print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)
# array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))

##### 원핫인코더(분류)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

y_train = y_train.reshape(-1,1) 
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_train.shape, y_test.shape) # (50000, 10) (10000, 10)

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(32, 32, 3))) 
model.add(Conv2D(32, (3,3), activation='relu' )) 
model.add(Dropout(0.3))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu' )) 
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation='relu' )) 
model.add(Conv2D(64, (3,3), activation='relu' ))  

model.add(Flatten()) 

model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=32, activation='relu'))

model.add(Dense(10, activation='softmax')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer='adam',
              metrics = ['acc'])

# es = EarlyStopping(
#     monitor='val_loss',
#     mode='min',
#     restore_best_weights=True,
#     verbose=1,
#     patience=200,
# )

# import datetime
# date = datetime.datetime.now()
# date = date.strftime('%d%m-%H%M')

# path ='./_save/keras36/'

# filename = '{epoch:04d}-{val_loss:.4f}.keras'
# filepath = "".join([path, "k36_", date, "-", filename])

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     save_best_only=True,
#     verbose=1,
#     filepath=filepath,
#     mode='min',
# )

start_time=time.time()
model.fit(x_train,y_train,
          epochs=50, 
          batch_size=128, 
          verbose=1,
          validation_split=0.2,
        #   callbacks = [es, mcp],
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


### 결과 1 (epoch : 10, batch_size : 128)
# loss :  1.06
# acc :  0.63
# acc_score :  0.6305
# 걸린시간 :  59.56 초

### 결과 2 (epoch : 30, batch_size : 64)
# loss :  1.03
# acc :  0.65
# acc_score :  0.6466
# 걸린시간 :  104.49 초

### 결과 3 
# loss :  1.76
# acc :  0.66
# acc_score :  0.6623
# 걸린시간 :  172.67 초

### 결과 4
# loss :  1.38
# acc :  0.67
# acc_score :  0.6677
# 걸린시간 :  203.82 초


