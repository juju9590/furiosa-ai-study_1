# 33-1 카피

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import time

path ='./_save/keras34/'


#1. 데이터
datasets = fetch_california_housing() 
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(20640, 8) (20640,)


x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.8,
    # test_size=0.2,
    # shuffle=True, #디폴트 섞는다
    random_state=777,
) 

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler() # 이상치에 강력


x_train = scaler.fit_transform(x_train) # 한줄로 사욯 가능
x_test = scaler.transform(x_test)    


print(np.min(x_train), np.max(x_train)) 
print(np.min(x_test), np.max(x_test)) 

#2. 모델구성
# model = Sequential()
# model.add(Dense(2, activation='relu', input_dim=8))
# model.add(Dropout(0.2))
# model.add(Dense(6, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(12))
# model.add(Dropout(0.5))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(1))

################# 함수형 모델
input1 = Input(shape=(8,))                                         # 입력층
dense1 = Dense(2, activation='relu', name='layer_1')(input1)        # 첫번째층
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(6, activation='relu', name='layer_2')(drop1)         # 두번째층
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(12, name='layer_3')(drop2)                           # 세번째층
drop3 = Dropout(0.5)(dense3)
dense4 = Dense(6, activation='relu', name='layer_4')(drop3)         # 네번째층
output1 = Dense(1)(dense4)

model = Model(inputs=input1, outputs=output1)                       # 모델정의


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
    verbose=1, #훈련중에 es가 진행되는지 볼 수 있음
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True, # 가장 최적의 가중치
    filepath=path + 'keras34_mcp1.keras', # 저장 파일명
    verbose=1, #훈련중에 mcp가 진행되는지 볼 수 있음
)

start_time = time.time()
hist = model.fit(x_train, 
                 y_train, 
                epochs=500, 
                batch_size=32,
                validation_split=0.2,
                callbacks = [es, mcp],
                verbose=1,                              
                )


# 훈련 중 가장 최적의 가중치를 자동으로 저장할 수 있도록 하는 장치 

end_time = time.time() # 현재시간을 반환, = 끝시간
print("걸린시간 :", round(end_time-start_time,2), "초")

print("=================== 학습 종료 ========================")

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



######## 결과 save

# loss(mse) : 0.37408486008644104
# r2 :  0.71
# rmse :  0.612

######## 결과 dropout
# r2 :  0.626
# mse :  0.482
# rmse :  0.695

######## 결과 함수형 모델 
# r2 :  0.554
# mse :  0.574
# rmse :  0.758

