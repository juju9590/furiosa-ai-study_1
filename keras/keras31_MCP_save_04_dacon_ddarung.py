# 28-4 카피

import numpy as np # 수치 계산에 특화
import pandas as pd # sklearn 만큼 강력함
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
path = "c:\study\_data\ddarung/" # 슬래시 역슬래시 / 2개 상관없음, 섞어쓰기 되지만 가급적 비권장

train_csv = pd.read_csv(path + "train.csv", index_col=0) # csv 가져오기, 인덱스 컬럼 데이터에 포함 x 
print(train_csv) # id열 포함 [1459 rows x 11 columns] -> id열 안 포함 [1459 rows x 10 columns] index_col=0 때문

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv) # [715 rows x 9 columns]

submission = pd.read_csv(path + "submission.csv", index_col=0)
print(submission) # [715 rows x 1 columns]

# 데이터 받으면 shape 찍기
print(train_csv.shape) # (1459, 10) // 훈련에서 분리해야 하는 데이터 train_test_split
print(test_csv.shape) # (715, 9) // 제출용 파일, y 없음
print(submission.shape) # (715, 1)

print(train_csv.columns) # 컬럼이 중요함

print(train_csv.info())
print(test_csv.info())

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



# 2. 모델 구성
model = Sequential()
model.add(Dense(8, input_dim=9))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    restore_best_weights=True,
    patience=50,
    verbose=1,
)

import datetime
date = datetime.datetime.now() 
print(date) 
print(type(date)) 
date = date.strftime("%m%d_%H%M") 
print(date) #
print(type(date)) 

path ='./_save/keras31/'

filename = '{epoch:04d}-{val_loss:.4f}.keras'
# d : int 형태, f : float 형태
filepath = "".join([path, "k31_04_", date, "-" ,filename])

 
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath,
    verbose=1,
)

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")

import time
start_time = time.time()
hist = model.fit(x_train, y_train, 
                epochs=500, batch_size=160,
                validation_split=0.2,
                callbacks = [es, mcp],
                verbose=1,
                )
end_time = time.time()

print("걸린시간 :", round(end_time-start_time,2), "초")

print("==============================================")

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 : ", round(r2,3))

mse = mean_squared_error(y_test, y_pred)
print("mse : ", round(mse,3))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", round(rmse,3))

############ 결과
# r2 :  0.679
# mse :  2311.922
# rmse :  48.082


