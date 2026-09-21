# 33-1 카피

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten,MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import time

# path ='./_save/keras30/'

#1. 데이터
datasets = fetch_california_housing() 
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(20640, 8) (20640,)

### train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.8,
    # test_size=0.2,
    # shuffle=True, #디폴트 섞는다
    random_state=777,
) 
### scaler (x값만 스케일링)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler() # 이상치에 강력

x_train = scaler.fit_transform(x_train) # 한줄로 사욯 가능
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) 
print(np.min(x_test), np.max(x_test)) 

### cnn 모델로 변환하기 위해 4차원으로 변환 (input_shape=(8,1,1))
x_train = x_train.reshape(-1,2,4,1)
x_test = x_test.reshape(-1,2,4,1)

print(x_train.shape, x_test.shape) #(16512, 2, 4, 1) (4128, 2, 4, 1)
print(y_train.shape, y_test.shape) #(16512,) (4128,)


#2. 모델구성
model = Sequential()

model.add(Conv2D(64, (2,2), padding='same', input_shape=(2, 4, 1))) 
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,1), padding='same', activation='relu' )) 
model.add(Conv2D(16, (1,1),padding='valid', activation='relu' )) 

# model.add(Flatten())
model.add(GlobalAveragePooling2D())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(1))  

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
    verbose=1, #훈련중에 es가 진행되는지 볼 수 있음
)

### MCP(ModelCheckpoint) : 모델 저장
import datetime
date = datetime.datetime.now() # 현재시간반환
# print(date) 
# print(type(date)) # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
# print(date) 
# print(type(date)) 

path ='./_save/keras30/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k30_", date, "-" ,filename])


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True, 
    filepath=filepath, 
    verbose=1, 
)

start_time = time.time()
hist = model.fit(x_train, 
                 y_train, 
                epochs=100, 
                batch_size=32,
                validation_split=0.2,
                # callbacks = [es, mcp],
                verbose=1,                              
                ) 

end_time = time.time() # 현재시간을 반환, = 끝시간


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, )
print("loss :", loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 : ", round(r2,3))

mse = mean_squared_error(y_test, y_pred)
print("mse : ", round(mse,3))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", round(rmse,3))

print("걸린시간 :", round(end_time-start_time,2), "초")




######################## 결과
# r2 :  0.72
# mse :  0.361
# rmse :  0.601

# epochs : 100, es, mcp 제외
############## GPU 속도
# 걸린시간 : 68.2 초

############## CPU 속도 ㅇ
# 걸린시간 : 40.02 초


##### dnn >> cnn 변경 1차
# loss : 0.29896432161331177
# r2 :  0.768
# mse :  0.299
# rmse :  0.547

##### dnn >> cnn 변경 2차
# r2 :  0.747
# mse :  0.326
# rmse :  0.571
# 걸린시간 : 125.83 초

##### dnn >> cnn 변경 2차 (CPU)
# r2 :  0.749
# mse :  0.323
# rmse :  0.568
# 걸린시간 : 90.08 초

