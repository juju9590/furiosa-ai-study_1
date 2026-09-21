# 33-8 카피

from sklearn.datasets import load_wine

## acc = 0.95 이상

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_wine()
# print(datasets.DESCR)
x = datasets['data']
y = datasets['target']
print(x.shape, y.shape) #(178, 13) (178,)

print(" =========== 원핫3. OneHotencoding() =============== ")

print(np.unique(y, return_counts=True)) 
#(array([0, 1, 2]), array([59, 71, 48])) 클래스가 3개 
print(y.shape) # (178,) 
y = y.reshape(-1, 1)
print(y.shape) # (178, 1) 

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False) 
y = ohe.fit_transform(y)
print(y, y.shape)

print("================ train_test_split ================== ")

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=999,
                                                    shuffle=True,
                                                    stratify=y,
                                                    )
print(x_train.shape, x_test.shape) # (124, 13) (54, 13)
print(y_train.shape, y_test.shape) # (124, 3) (54, 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) 
print(np.min(x_test), np.max(x_test))

########### X 데이터를 CNN에 넣기 위해 4차원으로 변환
x_train = x_train.reshape(-1,13,1,1)
x_test = x_test.reshape(-1,13,1,1)

print(x_train.shape, x_test.shape) #(142, 13, 1, 1) (36, 13, 1, 1)
print(y_train.shape, y_test.shape) #(142, 3) (36, 3)

# 2. 모델구성
model = Sequential()

model.add(Conv2D(64, (3,1), padding='same', input_shape=(13, 1, 1))) 
model.add(Conv2D(32, (2,1), padding='same', activation='relu' )) 
model.add(MaxPooling2D())

# model.add(Flatten())
model.add(GlobalAveragePooling2D())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(3, activation='softmax')) # 다중분류

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=500,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs=500, batch_size=128,
          verbose=1,
          validation_split=0.2,
          callbacks=[es],
          )
end_time = time.time()

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc :", result[1])

y_pred = np.argmax(model.predict(x_test), axis=1)
y_test = np.argmax(y_test, axis=1)
print(y_pred [:10])
print(y_test [:10])

acc_score = accuracy_score(y_test, y_pred)
print("acc_score : ", acc_score)
print("걸린시간 :", round((end_time-start_time),2),"초")


############################### RobustScaler 적용 후 ############################## ==> loss 개선, acc 하향

# Epoch 504/5000
# 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - acc: 1.0000 - loss: 3.2070e-07 - val_acc: 0.9310 - val_loss: 0.3985
# 걸린시간 : 30.12 초
# ===================== 학습 끝 ==========================
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - acc: 0.9167 - loss: 0.1447
# loss : 0.14471149444580078
# acc : 0.9166666865348816
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step
# [2 2 1 1 0 1 0 1 0 2]
# [2 2 1 1 0 1 0 1 0 2]
# acc_score :  0.9166666666666666

########## 드랍아웃  ==>향상
# loss : 0.09940765053033829
# acc : 0.9722222089767456
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 36ms/step
# [2 2 1 1 0 1 0 1 0 2]
# [2 2 1 1 0 1 0 1 0 2]
# acc_score :  0.9722222222222222

##### dnn >>> cnn (1차)
# loss : 0.042950477451086044
# acc : 0.9722222089767456
# acc_score :  0.9722222222222222
# 걸린시간 : 54.29 초

##### dnn >>> cnn (2차)
# loss :  0.2971
# acc :  0.8996
# acc_score :  0.8996
# 걸린시간 :  344.25 초



