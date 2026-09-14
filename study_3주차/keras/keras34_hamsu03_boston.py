#19-3 카피

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Dropout,Input
from tensorflow.keras.datasets import boston_housing
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np


# 텐서플로우에서 데이터셋을 가져올때 아래와 같이 가져오면 된다

(x_train, y_train), (x_test, y_test)= boston_housing.load_data()
print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
print(y_train.shape, y_test.shape) #(404,) (102,)

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

#2. 모델구성
# model = Sequential()
# model.add(Dense(5,input_dim=13))
# model.add(Dense(7, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(1))

################# 함수형 모델
input1 = Input(shape=(13,))                                         # 입력층
dense1 = Dense(7, activation='relu', name='layer_1')(input1)        
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(5, activation='relu', name='layer_2')(drop1)        
output1 = Dense(1)(dense2)

model = Model(inputs=input1, outputs=output1)                       # 모델정의




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')


import time
start_time = time.time()

hist = model.fit(x_train, y_train, 
          epochs=200, batch_size=64,
          validation_split=0.2,
          )
end_time = time.time()
print("걸린시간 :", round(end_time-start_time,2), "초")

# 훈련이 종료되면 마지막 W값이 정해진다.

print("=====================================")

#4. 평가, 성능
loss = model.evaluate(x_test, y_test, )
print(loss)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", rmse)

################################## 결과 ###################################
# Epoch 200/200
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 54.7659 - val_loss: 67.8529
# =====================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 67.0453 
# 67.04534149169922
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
# r2 :  0.19459107175796264
# rmse :  8.188121812667095

############################ MinMaxScaler 후 ########################  ===> 향상
# Epoch 200/200
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 39.4716 - val_loss: 41.4484
# =====================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 39.9959 
# 39.99590301513672
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
# r2 :  0.5195332540051056
# rmse :  6.324231259402225

############################ StandardScaler 후 ########################  ===> 향상
# Epoch 200/200
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 20.8262 - val_loss: 24.7887
# 걸린시간 : 11.3 초
# =====================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 27.6119
# 27.611854553222656
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
# r2 :  0.6683015685118041
# rmse :  5.25469827726185

############################ MaxAbsScaler 후 ########################  ===> loss 하향, R2/Rmse 향상
# Epoch 200/200
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 45.3736 - val_loss: 48.7276
# 걸린시간 : 11.03 초
# =====================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 48.1005 
# 48.100486755371094
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
# r2 :  0.4221737332224058
# rmse :  6.935451118881572

################### RobustScaler ##########################  ====> loss, R2, Rmse 성능개선

# Epoch 200/200
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 18.8596 - val_loss: 18.9100
# 걸린시간 : 11.13 초
# =====================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 21.9206 
# 21.92060661315918
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 
# r2 :  0.7366699474761296
# rmse :  4.681944788547605

############ 드랍아웃 적용 == 하향
# 39.65457534790039
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 
# r2 :  0.5236335361033014
# rmse :  6.297188083623483

############# 함수형 모델
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 99.6718 
# 99.67176818847656
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
# r2 :  -0.19734696673984464
# rmse :  9.983574987194404