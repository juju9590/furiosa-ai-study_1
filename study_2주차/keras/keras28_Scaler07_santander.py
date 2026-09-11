# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

# 22 카피


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
x = train_csv.drop(['target'], axis=1) 
y = train_csv['target']
print(x.shape, y.shape) #(200000, 200) (200000,)
print(np.unique(y, return_counts=True)) #(array([0, 1]), array([179902,  20098]))

x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size=0.7,
                 random_state=345,
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

print(np.min(x_train), np.max(x_train)) 
print(np.min(x_test), np.max(x_test))

print(x_train.shape, y_train.shape) #(140000, 200) (140000,)

#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=200, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['acc'],

              )

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    patience=50,
)

start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=1000,
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

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score : ", round(acc_score,4))

#1차 시도
# loss :  0.244
# acc :  0.9105
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 337us/step 
# acc_score :  0.9104666666666666

# ########### submission.csv 만들기 // target 컬럼에 넣어준다 #############
# # print(submission_csv)
# y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)

# submission_csv['target'] = y_submit
# print(submission_csv)

# submission_csv.to_csv(path + "submit/" + "submit_0908_1737.csv")

# ================================ 2차 시도 =======================================
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 581us/step - acc: 0.9105 - loss: 0.2454
# loss :  0.2454
# acc :  0.9105
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 317us/step 
# acc_score :  0.9105

# ================================ MinMaxScaler 적용 후  =========================== ==> 하향
# Epoch 100/100
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 250ms/step - acc: 0.9079 - loss: 0.2580 - val_acc: 0.9034 - val_loss: 0.2667
# 걸린시간 :  26.51 초
# ================================= 학습 종료 =================================
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 667us/step - acc: 0.9057 - loss: 0.2611
# loss :  0.2611
# acc :  0.9057
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 380us/step 
# acc_score :  0.9056

################################## StandardScaler 적용 후 ##################### ==> 향상
# Epoch 100/100
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 248ms/step - acc: 0.9217 - loss: 0.2134 - val_acc: 0.9062 - val_loss: 0.2527
# 걸린시간 :  26.35 초
# ================================= 학습 종료 =================================
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 656us/step - acc: 0.9089 - loss: 0.2468
# loss :  0.2468
# acc :  0.9089
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 387us/step
# acc_score :  0.9089

################################## StandardScaler + Batch_size & patience 재설정 ##################### ==> 향상
# Epoch 56/100
# 98/98 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.9548 - loss: 0.1250 - val_acc: 0.8809 - val_loss: 0.4542
# 걸린시간 :  13.44 초
# ================================= 학습 종료 =================================
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 573us/step - acc: 0.9102 - loss: 0.2456 
# loss :  0.2456
# acc :  0.9102
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 320us/step 
# acc_score :  0.9102

################################## MaxAbsScaler ##################### ==> loss 향상, acc 향상
# Epoch 66/100
# 98/98 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.9211 - loss: 0.2164 - val_acc: 0.9099 - val_loss: 0.2492
# 걸린시간 :  16.79 초
# ================================= 학습 종료 =================================
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 610us/step - acc: 0.9107 - loss: 0.2414 
# loss :  0.2414
# acc :  0.9107
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 334us/step 
# acc_score :  0.9106

################################## RobustScaler ##################### ==> 성능하향
# Epoch 56/100
# 98/98 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.9563 - loss: 0.1333 - val_acc: 0.8878 - val_loss: 0.3844
# 걸린시간 :  13.8 초
# ================================= 학습 종료 =================================
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 632us/step - acc: 0.9092 - loss: 0.2440 
# loss :  0.244
# acc :  0.9092
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 313us/step 
# acc_score :  0.9092