# 23-3 카피

from sklearn.datasets import fetch_covtype # 시간체크, 배치사이즈 크게 하기

# acc = 0.93 이상

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = fetch_covtype()
# print(datasets.DESCR)
'''
**Data Set Characteristics:**

=================   ============
Classes                        7
Samples total             581012
Dimensionality                54
Features                     int
=================   ============

:func:`sklearn.datasets.fetch_covtype` will load the covertype dataset;
it returns a dictionary-like 'Bunch' object
with the feature matrix in the ``data`` member
and the target values in ``target``. If optional argument 'as_frame' is
set to 'True', it will return ``data`` and ``target`` as pandas
data frame, and there will be an additional member ``frame`` as well.
The dataset will be downloaded from the web if necessary.
'''

x = datasets.data
y = datasets.target
# print(x.shape, y.shape) #(581012, 54) (581012,)

# print(np.unique(y, return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))


# print("================= 원핫1. to_categorical  =====================")
##########  to_categorical ###################
# from tensorflow.keras.utils import to_categorical 
# ### 카테고리컬로 하면 무조건 0부터 시작한다. 그래서 빈칸을 0으로 만들어 버린다. 그럼 0에 수렴하는 오류가 발생한다 
# ㄴ 이럴땐 0을 없애거나 추가 작업이 필요함. 그래서 지금 처럼 1~7 클래스가 나오면 판다스나 사이킷런 을 사용
# y = to_categorical(y)
# print(y)
# '''
# [[0. 0. 0. ... 1. 0. 0.]
#  [0. 0. 0. ... 1. 0. 0.]
#  [0. 0. 1. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]]
# '''
# print(y.shape) #(581012, 8)

print("================= 원핫2. pd.get_dummies()  =====================")

y = pd.get_dummies(y, dtype=int)
# print(y, y.shape)

'''
        1  2  3  4  5  6  7
0       0  0  0  0  1  0  0
1       0  0  0  0  1  0  0
2       0  1  0  0  0  0  0
3       0  1  0  0  0  0  0
4       0  0  0  0  1  0  0
...    .. .. .. .. .. .. ..
581007  0  0  1  0  0  0  0
581008  0  0  1  0  0  0  0
581009  0  0  1  0  0  0  0
581010  0  0  1  0  0  0  0
581011  0  0  1  0  0  0  0

[581012 rows x 7 columns] (581012, 7)
'''

# print("================= 원핫3. OneHotEncoding  =====================")
# from sklearn.preprocessing import OneHotEncoder
# print(np.unique(y), y.shape) # [1 2 3 4 5 6 7] (581012,)
# ㄴ y의 클래스는 7개, y의 쉐이프는 1차원으로 reshape로 변환 필요


# exit()


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=y,
                                                    )


from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()



scaler.fit(x_train) 

x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)    

# print(np.min(x_train), np.max(x_train)) 
# print(np.min(x_test), np.max(x_test))



#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=54, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
# learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(learning_rate=learning_rate), 
              metrics=['acc'],)

es = EarlyStopping(
                   monitor='val_loss',
                   mode='auto',
                   restore_best_weights=True,
                   patience=40,
                   )

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5, # learning_rate(러닝레이트) 비율 조절
)

start_time=time.time()
model.fit(x_train, y_train,
          epochs=500,
          batch_size=128,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, rlr],
          )
end_time=time.time()
print("걸린시간 :", round(end_time-start_time,3),"초")


#4. 예측, 평가
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc :", result[1])

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred )
print("acc_score :", acc_score)


####### RobustScaler 적용 후 
# 걸린시간 : 28.671 초
# loss : 0.5480256080627441
# acc : 0.7665206789970398
# acc_score : 0.7665206578143421


####### learning_rate = 0.01
# 걸린시간 : 52.289 초
# loss : 0.5555896162986755
# acc : 0.7685946226119995
# acc_score : 0.7685946145968693


# ReduceLROnPlateau
# 걸린시간 : 559.074 초
# loss : 0.5157950520515442
# acc : 0.7829918265342712
# acc_score : 0.7829918332573169

# learning_rate = 0.005, ReduceLR
# 걸린시간 : 862.433 초
# loss : 0.5012364387512207
# acc : 0.7852379083633423
# acc_score : 0.7852379026359044