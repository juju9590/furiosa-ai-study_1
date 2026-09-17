# 33-4 카피

import numpy as np # 수치 계산에 특화
import pandas as pd # sklearn 만큼 강력함
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
path = "D:\\Furiosa_AI\\study_3주차\\_data\\ddarung\\" # 절대경로
# path = "c:\study\_data\ddarung/" # 슬래시 역슬래시 / 2개 상관없음, 섞어쓰기 되지만 가급적 비권장


train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv) # [1459 rows x 11 columns] -> id열 안 포함 [1459 rows x 10 columns] index_col=0 때문

test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv) # [715 rows x 9 columns]

submission = pd.read_csv(path + "submission.csv", index_col=0)
# print(submission) # [715 rows x 1 columns]

# 데이터 받으면 shape 찍기
print(train_csv.shape) # (1459, 10) 
print(test_csv.shape) # (715, 9) 
print(submission.shape) # (715, 1)

# print(train_csv.columns) 

# print(train_csv.info())
# print(test_csv.info())

#### 결측치 처리
train_csv = train_csv.dropna() # 결측치 있는 row 자체를 없애버림
print(train_csv) # [1328 rows x 10 columns]

# train_csv를 x와 y로 분리
x = train_csv.drop(['count'], axis=1) # 열(컬럼) 삭제 | 행0, 열1
print(x) # [1328 rows x 9 columns]

y = train_csv['count'] # pandas에서 컬럼만 빼는거
print(y)
print(y.shape) # (1328,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=333)


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
x_train = x_train.reshape(-1,3,3,1)
x_test = x_test.reshape(-1,3,3,1)

print(x_train.shape, x_test.shape) #(929, 3, 3, 1) (399, 3, 3, 1)
print(y_train.shape, y_test.shape) #(929,) (399,)

# 2. 모델 구성
model = Sequential()

model.add(Conv2D(64, (2,1), input_shape=(3, 3, 1))) 
model.add(Conv2D(32, (2,1), padding='same', activation='relu' )) 
model.add(Conv2D(16, (1,1), padding='same', activation='relu' ))

# model.add(Flatten())
model.add(GlobalAveragePooling2D())

model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))

model.add(Dense(1,))  

model.summary()

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")

import time
start_time = time.time()
hist = model.fit(x_train, y_train, 
                epochs=500, batch_size=160,
                validation_split=0.2,
                )
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)
print("r2 : ", r2_score(y_test, y_pred))
print("rmse : ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("걸린시간 :", round(end_time-start_time,2), "초")

##### RobustScaler  ==>  loss 성능개선, R2, Rmse 개선
# 걸린시간 : 27.45 초
# loss :  2383.595947265625
# r2 :  0.6690272155320054
# rmse :  48.82208077283538


############ 드랍아웃 적용 ==> 하향
# loss :  4563.55810546875
# r2 :  0.36632983432447397
# rmse :  67.55410945306643

###### dnn >>> > cnn 1차
# loss :  1809.003662109375
# r2 :  0.7488118498116965
# rmse :  42.53238308731484

###### dnn >>> > cnn 2차
# 걸린시간 : 46.48 초
# loss :  2380.10791015625 (0에 가까울수록 좋음)
# r2 :  0.6695114740855965 (1에 가까울수록 좋음)
# rmse :  48.7863510002996 (0에 가까울수록 좋음)