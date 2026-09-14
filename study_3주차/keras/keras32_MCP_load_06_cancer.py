# 분류
# AI 모델은 분류(이진,다중 분류)와 회귀 모델 2가지만 있다.

# 21 카피

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.metrics import r2_score, mean_squared_error (회귀모델)

from sklearn.datasets import load_breast_cancer 

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR) 
print(datasets.feature_names) 

# x = datasets.data
x = datasets['data'] 
y = datasets.target 

print(x.shape, y.shape) #(569, 30) (569,)
print('타입 : ', type(x)) 

print(y) 

print("y의 범주 : ",np.unique(y)) # [0 1] 
print("y의 범주의 갯수 : ",np.unique(y, return_counts=True)) 
# (array([0, 1]), array([212, 357])) => 0은 212개, 1은 357개 

print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212
print(pd.Series(y).value_counts())
# 1    357
# 0    212

# 데이터셋 트레인,테스트 분리(7:3)
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    random_state=908,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    stratify=y, # (중요) y데이터를 스트레이트파이하란 이야기
                                                    # stratify = '층을 이루게 하다', '계층화하다', '계층별로 나누다'
                                                    )

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train) 
x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) 
print(np.min(x_test), np.max(x_test))

#2. 모델구성
path ='./_save/keras31/'
model = load_model(path + 'k31_06_0914_1444-0012-0.0458.keras')

# model = Sequential()
# model.add(Dense(30, input_dim=30, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(1, activation='sigmoid')) #(필수)이진분류모델

# #3.컴파일, 훈련
# model.compile(loss='binary_crossentropy', #(필수)이진분류모델
#               optimizer='adam', 
#               metrics=['accuracy'], 
#               ) 

# es = EarlyStopping(
#     monitor='val_loss',
#     mode='auto',
#     restore_best_weights=True,
#     patience=20,
#     verbose=1,
# )

# import datetime
# date = datetime.datetime.now() 
# print(date) 
# print(type(date)) 
# date = date.strftime("%m%d_%H%M") 
# print(date) #
# print(type(date)) 

# path ='./_save/keras31/'

# filename = '{epoch:04d}-{val_loss:.4f}.keras'
# filepath = "".join([path, "k31_06_", date, "-" ,filename])

 
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only=True,
#     filepath=filepath,
#     verbose=1,
# )

# start_time = time.time()
# model.fit(x_train, y_train,
#           epochs=100,
#           batch_size=32,
#           validation_split=0.3, #트레인에서 검증부분 분리
#           verbose=1,
#           callbacks=[es, mcp],
#           )
# end_time = time.time()
# print("걸린시간 :", round(end_time-start_time,2), "초")

# print("===================== 학습완료 ===========================")

#4. 성능, 평가
print("=========================================================")
loss = model.evaluate(x_test, y_test)
print("loss : ", round(loss[0],4)) # loss=binary_crossentropy
print("accuracy : ", round(loss[1],4))
print("=========================================================")

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) # 반올림 처리
 
from sklearn.metrics import accuracy_score 
acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", acc_score ) # acc_score :  0.9122807017543859



######## 결과 save
# loss :  0.1567
# accuracy :  0.9708
# =========================================================
# acc_score :  0.9707602339181286

######## 결과 load
# loss :  0.1567
# accuracy :  0.9708
# =========================================================
# acc_score :  0.9707602339181286

