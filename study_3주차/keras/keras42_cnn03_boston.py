#33-3 카피

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten,MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.datasets import boston_housing
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np

(x_train, y_train), (x_test, y_test)= boston_housing.load_data()
print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
print(y_train.shape, y_test.shape) #(404,) (102,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train) 

x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) 
print(np.min(x_test), np.max(x_test))

########### X 데이터를 CNN에 넣기 위해 4차원으로 변환
x_train = x_train.reshape(-1,13,1,1)
x_test = x_test.reshape(-1,13,1,1)

print(x_train.shape, x_test.shape) #
print(y_train.shape, y_test.shape) #


#2. 모델구성
model = Sequential()

model.add(Conv2D(64, (2,1), padding='same',  input_shape=(13, 1, 1))) 
model.add(Conv2D(32, (2,1) ,padding='same', activation='relu' )) 
model.add(Conv2D(32, (1,1) , activation='relu' )) 

# model.add(Flatten())
model.add(GlobalAveragePooling2D())

model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(1,))  

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

import time
start_time = time.time()

hist = model.fit(x_train, y_train, 
          epochs=200, batch_size=64,
          validation_split=0.2,
          )
end_time = time.time()

#4. 평가, 성능
loss = model.evaluate(x_test, y_test, )
print(loss)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", rmse)
print("걸린시간 :", round(end_time-start_time,2), "초")


##### RobustScaler ##########################  ====> loss, R2, Rmse 성능개선
# 걸린시간 : 11.13 초
# 21.92060661315918
# r2 :  0.7366699474761296
# rmse :  4.681944788547605

############ 드랍아웃 적용 == 하향
# 39.65457534790039
# r2 :  0.5236335361033014
# rmse :  6.297188083623483


####### dnn >>>> cnn 1차
# 22.854990005493164
# r2 :  0.7254452490629806
# rmse :  4.780689709382182

####### dnn >>>> cnn 2차
# 55.5850944519043
# r2 :  0.3322618159513461
# rmse :  7.455541311255175
# 걸린시간 : 18.47 초