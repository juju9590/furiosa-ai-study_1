# 23-2 카피


from sklearn.datasets import load_wine

## acc = 0.95 이상

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import time
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_wine()
# print(datasets.DESCR)
'''
**Data Set Characteristics:**

:Number of Instances: 178
:Number of Attributes: 13 numeric, predictive attributes and the class
:Attribute Information:
    - Alcohol
    - Malic acid
    - Ash
    - Alcalinity of ash
    - Magnesium
    - Total phenols
    - Flavanoids
    - Nonflavanoid phenols
    - Proanthocyanins
    - Color intensity
    - Hue
    - OD280/OD315 of diluted wines
    - Proline
    - class:
        - class_0
        - class_1
        - class_2
'''
x = datasets['data']
y = datasets['target']
print(x.shape, y.shape) #(178, 13) (178,)

############ 원핫인코딩 ###############
print(" =========== 원핫인코딩 to_categorical =============== ")
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# '''
# [0. 0. 1.]
#  [0. 0. 1.]
#  [0. 0. 1.]
#  [0. 0. 1.]
#  '''
# print(y.shape) #(178, 3)

print(" =========== 원핫2.  pd.get_dummies() =============== ")

# y = pd.get_dummies(y, dtype=int)
# print(y)

'''
     0  1  2
0    1  0  0
1    1  0  0
2    1  0  0
3    1  0  0
4    1  0  0
..  .. .. ..
173  0  0  1
174  0  0  1
175  0  0  1
176  0  0  1
177  0  0  1

[178 rows x 3 columns]
'''

print(" =========== 원핫3. OneHotencoding() =============== ")

print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48])) 클래스가 3개 
print(y.shape) # (178,) 1차원 형태.. 하지만 원핫인코딩은 2차원의 형태로 바꿔줘야 한다.
y = y.reshape(-1, 1)
print(y.shape) # (178, 1) 2차원(행렬=매트릭스) 형태로 바꿔줌

from sklearn.preprocessing import OneHotEncoder

# ohe = OneHotEncoder() #원핫 인코더 정의하고
# y = ohe.fit_transform(y)
# print(y, y.shape)

'''
Coords        Values
  (175, 2)      1.0
  (176, 2)      1.0
  (177, 2)      1.0 (178, 3)
'''

ohe = OneHotEncoder(sparse_output=False) # 매트릭스 형태로 바꿔주기 위해 sparse_output=False 파라미터 추가
y = ohe.fit_transform(y)
print(y, y.shape)

'''
[[1. 0. 0.]
...
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]] (178, 3)
 '''

# exit()

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

# scaler.fit(x_train) 
# x_train = scaler.transform(x_train)  
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) 
print(np.min(x_test), np.max(x_test))


# 2. 모델구성
path ='./_save/keras31/'
model = load_model(path + 'k31_08_0914_1456-0111-0.0733.keras')

# model = Sequential()
# model.add(Dense(30, input_dim=13, activation='relu'))
# model.add(Dense(120, activation='relu'))
# model.add(Dense(260, activation='relu'))
# model.add(Dense(80, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(3, activation='softmax')) # 다중분류

# # 3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', 
#               optimizer='adam', 
#               metrics=['acc'],
#               )

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


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
# filepath = "".join([path, "k31_08_", date, "-" ,filename])

 
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only=True,
#     filepath=filepath,
#     verbose=1,
# )

# start_time = time.time()
# model.fit(x_train, y_train,
#           epochs=5000, batch_size=16,
#           verbose=1,
#           validation_split=0.2,
#           callbacks=[es, mcp],
#           )

# end_time = time.time()
# print("걸린시간 :", round((end_time-start_time),2),"초")

# print("===================== 학습 끝 ==========================")
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


#### 결과 save
# loss : 0.07913465052843094
# acc : 0.9444444179534912

# acc_score :  0.9444444444444444

#### 결과 load
# loss : 0.07913465052843094
# acc : 0.9444444179534912

# acc_score :  0.9444444444444444

