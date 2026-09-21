# https://www.kaggle.com/competitions/bike-sharing-demand/overview
# 33-5 카피

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
path = "./_data/kaggle_bike/" # 학원
# path = "D:\\Furiosa_AI\\study_3주차\\_data\\kaggle_bike\\" # 개인

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

########### X 데이터를 CNN에 넣기 위해 4차원으로 변환
x_train = x_train.reshape(-1,4,2,1)
x_test = x_test.reshape(-1,4,2,1)

print(x_train.shape, x_test.shape) #(8708, 4, 2, 1) (2178, 4, 2, 1)
print(y_train.shape, y_test.shape) #(8708,) (2178,)

# exit()

# 2. 모델 구성
model = Sequential()

model.add(Conv2D(64, (2,1), padding='same', input_shape=(4, 2, 1))) 
model.add(Conv2D(32, (2,1), padding='same', activation='relu' ))
model.add(MaxPooling2D()) 

# model.add(Flatten())
model.add(GlobalAveragePooling2D())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(1,))  

model.summary()

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")

import time
start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=100, batch_size=32,
                 validation_split=0.2,
          )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", rmse)
print("걸린시간 :", round(end_time-start_time,2), "초")

####### RobustScaler 
# 걸린시간 : 22.2 초
# loss :  22406.55078125
# r2 :  0.2988594174385071
# rmse :  149.68817197135184

########### 드롭아웃 ===> 성능하향
# loss :  25349.328125
# r2 :  0.20677465200424194
# rmse :  159.21472946346705


####### dnn >>>> cnn 1차
# loss :  21573.443359375
# r2 :  0.32492876052856445
# rmse :  146.87900925379023

####### dnn >>>> cnn 2차
# loss :  24145.662109375
# r2 :  0.24443942308425903
# rmse :  155.38875768737262
# 걸린시간 : 63.71 초