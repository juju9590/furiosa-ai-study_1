# 33-5
# https://www.kaggle.com/competitions/bike-sharing-demand/overview

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
path = "./_data/kaggle_bike/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# 데이터 확인
print(train_csv) # [10886 rows x 11 columns]
print(test_csv) # [6493 rows x 8 columns]
print(submission) # [6493 rows x 1 columns]

# shape 확인
print(train_csv.shape) # (10886, 11)
print(test_csv.shape) # (6493, 8)
print(submission.shape) # (6493, 1)

# 결측치 확인
print(train_csv.info()) # 결측치 X
print(test_csv.info()) # 결측치 X

print(train_csv.describe()) # 이상치 확인 가능 (봄~겨울 1~4)

print(train_csv.isna().sum()) # 컬럼별 결측치 확인
print(test_csv.isnull().sum()) # 컬럼별 결측치 확인

########### x, y 분리 #############3

x = train_csv.drop(['casual', 'registered', 'count'], axis=1) #열(컬럼) 삭제 
print(x) # [10886 rows x 8 columns]

y = train_csv['count']
print(y) # (10886,)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=142,
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


# 2. 모델 구성
# model = Sequential()
# model.add(Dense(12, activation='relu', input_dim=8))
# model.add(Dropout(0.5))
# model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(1))

################# 함수형 모델
input1 = Input(shape=(8,))                                         # 입력층
dense1 = Dense(12, activation='relu', name='layer_1')(input1) 
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(8, activation='relu', name='layer_2')(drop1)      
drop2 = Dropout(0.5)(dense2)
dense3 = Dense(4, activation='relu', name='layer_3')(drop2)  
output1 = Dense(1)(dense3)

model = Model(inputs=input1, outputs=output1)                       # 모델정의


# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")

import time
start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=100, batch_size=32,
                 validation_split=0.2,
          )
end_time = time.time()

print("걸린시간 :", round(end_time-start_time,2), "초")


print("================== 학습 종료 ===================")

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", rmse)


##################################### 결과 ###########################
# Epoch 100/100
# 218/218 ━━━━━━━━━━━━━━━━━━━━ 0s 884us/step - loss: 23428.4336 - val_loss: 24937.2461
# =================================================
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 674us/step - loss: 23075.4004
# loss :  23075.400390625
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 890us/step
# r2 :  0.27792996168136597
# rmse :  151.9058867769778

##################################### MinMaxScaler 적용  ############## ==> 소폭 향상
# Epoch 100/100
# 218/218 ━━━━━━━━━━━━━━━━━━━━ 0s 916us/step - loss: 23041.4199 - val_loss: 24915.9121
# =================================================
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 714us/step - loss: 22969.3359
# loss :  22969.3359375
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 715us/step
# r2 :  0.28124868869781494
# rmse :  151.55639162948555

##################################### StandardScaler 적용  ############## ==> 소폭 향상
# Epoch 100/100
# 218/218 ━━━━━━━━━━━━━━━━━━━━ 0s 879us/step - loss: 22292.0059 - val_loss: 24245.3984
# 걸린시간 : 22.26 초
# ================== 학습 종료 ===================
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 712us/step - loss: 22668.6934
# loss :  22668.693359375
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 725us/step
# r2 :  0.29065656661987305      ### 1에 가까울수록 좋고
# rmse :  150.56125466483732     ### 0에 가까울수록 좋다.

##################################### MaxAbsScaler 적용  ############## ==> loss 하향, r2, rmse 향상
# Epoch 100/100
# 218/218 ━━━━━━━━━━━━━━━━━━━━ 0s 902us/step - loss: 23191.3379 - val_loss: 24926.5566
# 걸린시간 : 22.09 초
# ================== 학습 종료 ===================
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 666us/step - loss: 23043.7012
# loss :  23043.701171875
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 786us/step
# r2 :  0.27892178297042847
# rmse :  151.80152543699947

##################################### RobustScaler 적용  ############## ==> 성능개선

# Epoch 100/100
# 218/218 ━━━━━━━━━━━━━━━━━━━━ 0s 950us/step - loss: 22467.1953 - val_loss: 24203.8477
# 걸린시간 : 22.2 초
# ================== 학습 종료 ===================
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 655us/step - loss: 22406.5508
# loss :  22406.55078125
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 861us/step
# r2 :  0.2988594174385071
# rmse :  149.68817197135184

########### 드롭아웃 ===> 성능하향
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 753us/step - loss: 25349.3281
# loss :  25349.328125
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 844us/step
# r2 :  0.20677465200424194
# rmse :  159.21472946346705

########### 함수형 모델
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 744us/step - loss: 24240.1719
# loss :  24240.171875
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 759us/step
# r2 :  0.2414821982383728
# rmse :  155.69254934605894