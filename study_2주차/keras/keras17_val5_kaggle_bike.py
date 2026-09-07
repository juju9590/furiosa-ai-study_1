# 14-1 카피


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

# 2. 모델 구성
model = Sequential()
model.add(Dense(12, activation='relu', input_dim=8))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.33,
          )

print("=================================================")

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", rmse)

########### submission.csv 만들기 // count 컬럼에 넣어준다 #############
print(submission)
y_submit = model.predict(test_csv)

submission['count'] = y_submit
print(submission)

submission.to_csv(path + "submit/" + "submit_0907_13:36.csv")

"""
random : 142
train_size : 0.8
epochs = 300
batch_size = 32
model.add(Dense(12, activation='relu', input_dim=8))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))
================================
loss :  21874.169921875
r2 :  0.3155185580253601
rmse :  147.8991883746324
"""

# Epoch 299/300
# 183/183 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 51255.8711 - val_loss: 54267.5742
# Epoch 300/300
# 183/183 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 51207.0742 - val_loss: 54217.9766
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 619us/step - loss: 50548.8320
# loss :  50548.83203125
# 69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 732us/step
# r2 :  -0.5817623138427734
# rmse :  224.8306741333353
