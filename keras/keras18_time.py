# 14-1 copy

# https://www.kaggle.com/competitions/bike-sharing-demand/overview

# 활성 함수
# Linear
# relu : 0이상은 그대로, 음수는 0처리

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time 

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

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.9, random_state=142
                                                    )

# 2. 모델 구성
model = Sequential()
model.add(Dense(12, activation='relu', input_dim=8))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")

start_time = time.time() #현재시간을 반환, = 시작시간
model.fit(x_train, y_train, epochs=10, batch_size=8)
end_time = time.time() # 현재시간을 반환, = 끝시간

print("===============================================")

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", rmse)

print("걸린시간 :", round(end_time - start_time,2), "초")

# Epoch 10/10
# 1225/1225 ━━━━━━━━━━━━━━━━━━━━ 1s 587us/step - loss: 24119.0527
# 35/35 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 24536.0312 
# loss :  24536.03125
# 35/35 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step 
# r2 :  0.25890225172042847
# rmse :  156.63981997922815
# 걸린시간 : 8.41 초 (epochs=10, batch_size=8)


'''
########### submission.csv 만들기 // count 컬럼에 넣어준다 #############
print(submission)
y_submit = model.predict(test_csv)

submission['count'] = y_submit
print(submission)

submission.to_csv(path + "submit/" + "submit_0907_1059.csv")

'''
