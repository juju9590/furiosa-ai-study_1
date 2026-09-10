# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

path = 'c:/study/_data/kaggle_santander/' #절대경로
# path = './_data/kaggle_santander/' #상대경로

train_csv = pd.read_csv(path + "train.csv", index_col=0) #파일이 있는 경로 표시
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape) #(200000, 201)
print(test_csv.shape) #(200000, 200)
print(submission_csv.shape) #(200000, 1)

# 결측치 보기

# print(train_csv.info()) # 1
print(train_csv.isna().sum()) #2
print(test_csv.isnull().sum()) #3


# 데이터
x = train_csv.drop(['target'], axis=1) #트레인 데이터에서
y = train_csv['target']
print(x.shape, y.shape) #(200000, 200) (200000,)
print(np.unique(y, return_counts=True)) #(array([0, 1]), array([179902,  20098]))

print("====================== 원핫2 ========================")

# y = pd.get_dummies(y, dtype=int)
# print(y, y.shape ) # (200000, 2)

'''
              0  1
ID_code           
train_0       1  0
train_1       1  0
train_2       1  0
train_3       1  0
train_4       1  0
...          .. ..
train_199995  1  0
train_199996  1  0
train_199997  1  0
train_199998  1  0
train_199999  1  0 [200000 rows x 2 columns] (200000, 2)

'''

print("====================== 원핫3 ========================")
from sklearn.preprocessing import OneHotEncoder

print(y, y.shape )

# y기 시리즈 형태로 reshape 안됨, 넘파이로 바꾼 후 reshape 하면 됨
# y = np.array(y) 
y = y.to_numpy(y)
y= y.reshape(-1, 1)
print(y, y.shape)

'''
[[0]
 [0]
 [0]
 ...
 [0]
 [0]
 [0]] (200000, 1)

'''

ohe = OneHotEncoder(sparse_output=False) 
y = ohe.fit_transform(y)
print(y, y.shape)

'''
[[1. 0.]
 [1. 0.]
 [1. 0.]
 ...
 [1. 0.]
 [1. 0.]
 [1. 0.]] (200000, 2)
 '''

# exit()


x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size=0.8,
                 random_state=345,
                 stratify=y,

                 )

print(x_train.shape, y_train.shape) #(140000, 200) (140000,)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=200, activation='relu'))
model.add(Dense(160, activation='relu'))
model.add(Dense(280, activation='relu'))
model.add(Dense(140, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['acc'],

              )

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    patience=500,
)

start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=10000,
    verbose=1,
    validation_split=0.3,
    callbacks=[es],
)
end_time = time.time()
print("걸린시간 : ", round((end_time - start_time),2),"초") 

print("================================= 학습 종료 =================================")

#4. 성능, 평가
loss = model.evaluate(x_test, y_test)
print("loss : ", round(loss[0],4))
print("acc : ", round(loss[1],4))

y_pred = np.argmax(model.predict(x_test), axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score : ", round(acc_score,4))


# ================================ 이진분류 결과 =====================================
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 581us/step - acc: 0.9105 - loss: 0.2454
# loss :  0.2454
# acc :  0.9105
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 317us/step 
# acc_score :  0.9105


# ================================ 다중분류 결과 ======================================
# Epoch 100/100
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - acc: 0.9133 - loss: 0.2339 - val_acc: 0.9038 - val_loss: 0.2677
# 걸린시간 :  26.74 초
# ================================= 학습 종료 =================================
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 773us/step - acc: 0.9075 - loss: 0.2529
# loss :  0.2529
# acc :  0.9075
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 560us/step
# acc_score :  0.9075


# ============================================================
# Epoch 945/1000
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - acc: 0.9750 - loss: 0.0808 - val_acc: 0.8752 - val_loss: 0.9920
# 걸린시간 :  261.49 초
# ================================= 학습 종료 =================================
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 783us/step - acc: 0.9083 - loss: 0.2489
# loss :  0.2489
# acc :  0.9083
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 504us/step
# acc_score :  0.9083

# Epoch 607/1000
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - acc: 1.0000 - loss: 5.0230e-06 - val_acc: 0.8705 - val_loss: 3.4839
# 걸린시간 :  158.14 초
# ================================= 학습 종료 =================================
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 750us/step - acc: 0.9081 - loss: 0.2539
# loss :  0.2539
# acc :  0.9081
# 1250/1250 ━━━━━━━━━━━━━━━━━━━━ 1s 532us/step
# acc_score :  0.9081