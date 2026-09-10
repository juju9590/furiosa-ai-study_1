# 23-1 copy

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

#1. 데이터
datesets = load_iris()
print(datesets.DESCR) 

print(datesets.feature_names) 
x = datesets['data'] #(150, 4)
y = datesets['target'] #(150,)

print(x.shape)
print(y.shape)
print(y)
print(np.unique(y, return_counts=True)) 

########################## 원핫1. to_categorical ####################################
# to_categorical 함수를 사용하여 원핫인코딩 형태로 바꿔준다
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)

########################## 원핫2. pandas ####################################
# pandas → get_dummies()
# print(" =========== 원핫2 =============== ")
# y = pd.get_dummies(y) 
# print(y)

# y = pd.get_dummies(y, dtype=int) # 정수로
# print(y) 

########################## 원핫3. sklearn ####################################
# 3. scikit learn → OneHotEncoder

# print(" =========== 원핫3. OneHotEncoder  =============== ")

# from sklearn.preprocessing import OneHotEncoder # 전처리

# ohe = OneHotEncoder() # 원핫인코더 정의
# 사이킷런의 원핫인코더는 매트릭스(=행렬) 형태의 자료를 원함

# y = ohe.fit_transform(y)
# print(y)
from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(150,1) #  (150, 1)
y = y.reshape(-1,1) #  (150, 1), 행의 가장 마지막 자리 모를 경우 -1로 기재 (현재 마지막이 150이라 150 나옴)
print(y, y.shape) # (150, 1)


# ohe = OneHotEncoder() # sparse 형태로 나온다.

ohe = OneHotEncoder(sparse_output=False) # 매트릭스(=행렬) 형태의 자료를 원함 => sparse_output=False 추가 

y = ohe.fit_transform(y)
print(y, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=333,
                                                    shuffle=True, # 나중에 시계열 빼고, 셔플은 모두 True 로 할 것
                                                    stratify=y, # 편향되지 않게 추출하기 위해 꼭 y로 설정 
                                                    ) 

print(x_train.shape, y_train.shape) #(120, 4) (120, 3)
print(x_test.shape, y_test.shape) # (30, 4) (30, 3)

#2. 모델구성
model = Sequential()
# model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, input_shape=(4,), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, )) 
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax')) 

'''
차원    |  원 데이터      |  input_shape
--------------------------------------------------
2차원   (n,4)            => input_shape(4,)
3차원   (n,100,3)        => input_shape(100,3)
4차원   (n,100,100,3)    => input_shape(100,100,3)
--------------------------------------------------
'''

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['acc'],              
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)

start_time= time.time()
model.fit(x_train, y_train,
          epochs=1000,
          batch_size=16,
          verbose=1,
          validation_split=0.2,
          callbacks=[es],
          )
end_time= time.time()

# ======================== 훈련종료 =====================================

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print("loss :", round(result[0],3))
print("acc :", round(result[1],3))

y_pred = model.predict(x_test) 
print(y_pred) 

y_pred = np.argmax(y_pred, axis=1) 
print(y_pred)

print(y_test) 

y_test = np.argmax(y_test, axis=1) 
print(y_test)

accuracy_score = accuracy_score(y_test, y_pred)
print("acc_score :", round(accuracy_score,3))
print("걸린시간 : ", round(end_time - start_time, 2),"초")
