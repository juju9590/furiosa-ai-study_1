#19-3 카피

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np

path ='./_save/keras31/'


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
model = Sequential()
model.add(Dense(5,input_dim=13))
model.add(Dense(7, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
    verbose=1,
)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
path ='./_save/keras31/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path,"K31",date,"-", filename ])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=filepath,
    verbose=1,

)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

import time
start_time = time.time()

hist = model.fit(x_train, y_train, 
          epochs=200, batch_size=64,
          validation_split=0.2,
          callbacks=[es,mcp],
          verbose=1,
          )
end_time = time.time()
print("걸린시간 :", round(end_time-start_time,2), "초")


print("=====================================")

#4. 평가, 성능
loss = model.evaluate(x_test, y_test, )
print(loss)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ", round(r2,3))

mse = mean_squared_error(y_test, y_pred)
print("mse : ", round(mse,3))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", round(rmse,3))

############### 결과 (Save)
# r2 :  0.686
# mse :  26.146
# rmse :  5.113
