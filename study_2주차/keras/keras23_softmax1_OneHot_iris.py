# 다중분류 
# y의 클래스 값 확인 필수
# 다중분류 : OneHotencoding -> 백터화(to_categorical) -> train_test_split -> 
# (왜? 통상적으로 원핫하고 카테고리화 하고 트레인, 테스트 분리해준다.) 
# 원핫인코딩의 치명적 문제 ==> 클래스가 많으면 0이 너무 많다. 의미없는 데이터가 많아짐. => 성능저하

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
# print(datesets)
print(datesets.DESCR) ### 판다스 디스크라이브 
# Instances = low = 행
# Attributes = 컬럼 = 열 = 피치 = 속성 = 특성

# Iris plants dataset
# --------------------

# **Data Set Characteristics:**

# :Number of Instances: 150 (50 in each of three classes)
# :Number of Attributes: 4 numeric, predictive attributes and the class
# :Attribute Information:
#     - sepal length in cm
#     - sepal width in cm
#     - petal length in cm
#     - petal width in cm
#     - class:                       ### y의 클래스가 3가지
#             - Iris-Setosa
#             - Iris-Versicolour
#             - Iris-Virginica

# :Summary Statistics:

# ============== ==== ==== ======= ===== ====================
#                 Min  Max   Mean    SD   Class Correlation
# ============== ==== ==== ======= ===== ====================
# sepal length:   4.3  7.9   5.84   0.83    0.7826
# sepal width:    2.0  4.4   3.05   0.43   -0.4194
# petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
# petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
# ============== ==== ==== ======= ===== ====================

# :Missing Attribute Values: None
# :Class Distribution: 33.3% for each of 3 classes.
# :Creator: R.A. Fisher
# :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
# :Date: July, 1988

print(datesets.feature_names) #### 판다스에서 컬럼스 
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = datesets['data'] #(150, 4)
y = datesets['target'] #(150,)

print(x.shape)
print(y.shape)
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([50, 50, 50]))
# 판다스에서 value.count()로 가능

'''
[0,0,1,1,2] #(5,)  숫자로 보여지는 것을 원핫인코딩하여 값이 없는 위치값으로 변경한다.
-> 
[[1,0,0]
[1,0,0]
[0,1,0]
[0,1,0]
[0,1,0]]   #(5,3)
  
'''
########################## 원핫1. to_categorical ####################################
# to_categorical 함수를 사용하여 원핫인코딩 형태로 바꿔준다
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)

'''
(150,)
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]


(array([0, 1, 2]), array([50, 50, 50]))
[[1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
...
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 1.]] #(150, 3)

'''
########################## 원핫2. pandas ####################################
# pandas → get_dummies()
print(" =========== 원핫2 =============== ")
# y = pd.get_dummies(y) 
# print(y)
'''
불리언
         0      1      2
0     True  False  False
1     True  False  False
2     True  False  False
3     True  False  False
4     True  False  False
..     ...    ...    ...
145  False  False   True
146  False  False   True
147  False  False   True
148  False  False   True
149  False  False   True
'''
# y = pd.get_dummies(y, dtype=int) # 정수로
# print(y)
'''
     0  1  2
0    1  0  0
1    1  0  0
2    1  0  0
3    1  0  0
4    1  0  0
..  .. .. ..
145  0  0  1
146  0  0  1
147  0  0  1
148  0  0  1
149  0  0  1
'''

# exit()

########################## 원핫3. sklearn ####################################
# 3. scikit learn → OneHotEncoder

print(" =========== 원핫3. OneHotEncoder  =============== ")

# from sklearn.preprocessing import OneHotEncoder # 전처리

# ohe = OneHotEncoder() # 원핫인코더 정의
# 사이킷런의 원핫인코더는 매트릭스(=행렬) 형태의 자료를 원함

# y = ohe.fit_transform(y)
# print(y)
'''
ValueError: Expected 2D array, got 1D array instead:
array=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2].
Reshape your data either using array.reshape(-1, 1) 
if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

< reshape 의 조건 > 
1. 내용(값)이 바뀌면 안됨
2. 순서가 바뀌면 안됨
예) 1차원 (3,) = [1,2,3]     ->  2차원 (3,1) = [[1],[2],[3]]

나중에 CNN에서는 모든 데이터에서 reshape가 사용된다.
내용과 순서가 바뀌지 않으면 어떤 모양(1, 2, 3, 4 차원...) 으로도 바뀔 수 있다!!!

'''
from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(150,1) #  (150, 1)
y = y.reshape(-1,1) #  (150, 1), 행의 가장 마지막 자리 모를 경우 -1로 기재 (현재 마지막이 150이라 150 나옴)
print(y, y.shape) # (150, 1)

# exit()
# ohe = OneHotEncoder() # sparse 형태로 나온다.

ohe = OneHotEncoder(sparse_output=False) # 매트릭스(=행렬) 형태의 자료를 원함 => sparse_output=False 추가 

y = ohe.fit_transform(y)
print(y, y.shape)

# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=333,
                                                    shuffle=True, # 나중에 시계열 빼고, 셔플은 모두 True 로 할 것
                                                    stratify=y, # 편향되지 않게 추출하기 위해 꼭 y로 설정 
                                                    ) 

print(x_train.shape, y_train.shape) #(120, 4) (120, 3)
print(x_test.shape, y_test.shape) # (30, 4) (30, 3)


# exit()

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, )) # 중간에 활성화 함수 없어도 상관없다. 
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax')) # 다중분류는 마지막 레이어에서는 무조건 (softmax는 모든 행의 합이 1이 넘지 않도록 해준다.)
# y에서 추출된 값을 모두 더한 후 N 빵 하기 때문에 1을 넘길 수 없다. 
# 큰수는 더 확실하게 커지고, 작은 수는 더 작아지게 만들어 argmax 처리 시 분명하게 구분할 수 있도록 해준다
# https://wikidocs.net/35476

# 숫자의 값이 아니라 위치로 변경하여 인식 시킨다 
# ㄴ = OneHotEncoding
# 여자분 0 [1,0,0] 
# 남자분 1 [0,1,0]
# 외계인 2 [0,0,1]

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', # 다중분류에서 무조건 
              # https://wordbe.tistory.com/46 크로스엔트로피,, 수식에서 0과  1을 넣으면 한쪽이 날라가는 수식
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

y_pred = model.predict(x_test) # 예측 or 추론, (y_pred = W* x_test + b)
print(y_pred) 
'''
[[9.99763310e-01 2.36683321e-04 4.59430751e-08]
 [4.89274470e-08 2.43033702e-03 9.97569621e-01]
 [9.99942780e-01 5.72124118e-05 5.21152455e-09]]

'''
y_pred = np.argmax(y_pred, axis=1) # y예측값 변환 (추출된 소프트맥스값은 실수로 -> [1,0,0],[0,1,0],[0,0,1] 형태로 바꿔주기 위해)
print(y_pred)
# [0 2 0 2 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 2]

print(y_test) # 데이터 부분에서 원핫인코딩 했기때문에 다시 비교할 수 있는 값으로 변경 필요

'''
[[1. 0. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]  ... ]
'''
y_test = np.argmax(y_test, axis=1) # y원값 변환 (1,2,3 -> [1,0,0],[0,1,0],[0,0,1] 형태로 바꿔주기 위해)
print(y_test)
# [0 2 0 1 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1]
# ㄴ ㄴ ㄴ y_pred 와 y_test를 같은 형태로 바꿔줘야 비교가 되는거기 때문에 np.argmax() 처리 필요

accuracy_score = accuracy_score(y_test, y_pred)
print("acc_score :", round(accuracy_score,3))
print("걸린시간 : ", round(end_time - start_time, 2),"초")

# ============================= 결과 1 ===================================

# Epoch 1000/1000
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - acc: 0.9792 - loss: 0.0485 - val_acc: 0.9583 - val_loss: 0.0420
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - acc: 0.9667 - loss: 0.0856
# loss : 0.086
# acc : 0.967
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 49ms/step
# acc_score : 0.967
# 걸린시간 :  55.14 초

# ============================= 결과 2 ===================================
# Epoch 421/1000
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 1s 98ms/step - acc: 0.9688 - loss: 0.0438 - val_acc: 1.0000 - val_loss: 0.0623
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 169ms/step - acc: 0.9667 - loss: 0.0799
# loss : 0.08
# acc : 0.967
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 179ms/step
# acc_score : 0.967
# 걸린시간 :  212.73 초






