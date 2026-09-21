# https://www.kaggle.com/competitions/bike-sharing-demand/overview

# 활성 함수
# Linear
# relu : 0이상은 그대로, 음수는 0처리

# 19-5  카피

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
model = Sequential()
model.add(Dense(12, activation='relu', input_dim=8))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

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


# print("================== history ===============================")
# print(hist)
# print("================== hist.history ==========================")
# print(hist.history)
# print("================== loss ==========================")
# print(hist.history['loss'])
# print("================== val_loss ==========================")
# print(hist.history['val_loss'])

# print("================== 시각화 ==========================")
# import matplotlib.pyplot as plt

# # plt에서 한글 못 읽기 때문에 맑은고딕 폰트 설정 필수
# # plt.rcParams['font.family']='Malgun Gothic'
# plt.rc('font', family = 'Hancom Gothic')

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# # x를 명시하지 않으면 y값을 시간순으로 그려줌
# plt.legend(loc="upper right") # 우측 상단에 라벨표시(범례)
# plt.title('캐글 Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid() #모눈종이처럼 표시(격자)
# plt.show()

# 최소의 오차(loss), 최적의 웨이트


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



