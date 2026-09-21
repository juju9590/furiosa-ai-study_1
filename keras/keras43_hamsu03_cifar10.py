# 41-3 copy


# 실습 acc : 0.67 (n, 32,32,3)

from tensorflow.keras.datasets import cifar10

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Input
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

x_train = x_train.reshape(-1,32*32*3)
x_test = x_test.reshape(-1,32*32*3)
# print(x_train.shape, x_test.shape) # (50000, 3072) (10000, 3072)

# exit()

##### y값 알아보기
# print(np.unique(y_train, return_counts=True))
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
# model = Sequential()
# model.add(Dense(512, input_shape=(32*32*3,), activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))

# model.add(Dense(10, activation='softmax')) 

##### 함수형 모델 
input1 = Input(shape=(32*32*3, ))
danse1 = Dense(512, activation='relu' )(input1)
danse2 = Dense(512, activation='relu' )(danse1)
danse3 = Dense(512, activation='relu' )(danse2)
drop1 = Dropout(0.2)(danse3)
danse4 = Dense(256, activation='relu' )(drop1)
danse5 = Dense(256, activation='relu' )(danse4)
danse6 = Dense(256, activation='relu' )(danse5)
drop2 = Dropout(0.2)(danse6)
danse7 = Dense(128, activation='relu' )(drop2)
danse8 = Dense(128, activation='relu' )(danse7)
danse9 = Dense(128, activation='relu' )(danse8)
drop3 = Dropout(0.2)(danse9)
danse10 = Dense(64, activation='relu' )(drop3)
danse11 = Dense(64, activation='relu' )(danse10)
danse12 = Dense(64, activation='relu' )(danse11)
drop4 = Dropout(0.2)(danse12)
danse13 = Dense(32, activation='relu' )(drop4)
danse14 = Dense(32, activation='relu' )(danse13)
drop5 = Dropout(0.2)(danse14)
danse15 = Dense(16, activation='relu' )(drop5)
danse16 = Dense(16, activation='relu' )(danse15)

output1 =Dense(10, activation='softmax')(danse16)

model = Model(inputs=input1, outputs=output1)
model.summary()

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer='adam',
              metrics = ['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    verbose=1,
    patience=50,
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
          epochs=100, 
          batch_size=2048, 
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

### CNN
### GlobalAveragePooling2D
# loss :  0.67
# acc :  0.77
# acc_score :  0.7719
# 걸린시간 :  275.09 초

### DNN 3차
# loss :  1.6
# acc :  0.44
# acc_score :  0.4441
# 걸린시간 :  55.33 초

### DNN 4차 
# loss :  1.85
# acc :  0.47
# acc_score :  0.4713
# 걸린시간 :  35.14 초

### DNN 4차 
# loss :  1.95
# acc :  0.47
# acc_score :  0.4684
# 걸린시간 :  25.51 초

