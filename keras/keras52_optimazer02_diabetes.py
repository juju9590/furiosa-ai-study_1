# 19-2 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

from tensorflow.keras.callbacks import EarlyStopping


# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442, )

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                train_size=0.8, 
                                random_state=333,
                                
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


# 2. 모델 구성
model = Sequential()
model.add(Dense(30, input_dim=10, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
# learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
# learning_rate = 0.005
learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

import time
start_time = time.time()

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    restore_best_weights=True,
    patience=50,

)

hist = model.fit(x_train, y_train, 
          epochs=500, 
          batch_size=64,
          validation_split=0.20,
          callbacks = [es],
          )

end_time = time.time()
print("걸린시간 :", round(end_time-start_time,2), "초")


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", rmse)

# print("================== history ===============================")
# print(hist)
# print("================== hist.history ==========================")
# print(hist.history)
# print("================== loss ==========================")
# print(hist.history['loss'])
# print("================== val_loss ==========================")
# print(hist.history['val_loss'])

# print("================== 시각화 ==========================")
# import matplotlib.pyplot as plt

# # plt에서 한글 못 읽기 때문에 맑은고딕 폰트 설정 필수
# # plt.rcParams['font.family']='Malgun Gothic'
# plt.rc('font', family = 'Hancom Gothic')

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# # x를 명시하지 않으면 y값을 시간순으로 그려줌
# plt.legend(loc="upper right") # 우측 상단에 라벨표시(범례)
# plt.title('diabetes 당뇨병 Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid() #모눈종이처럼 표시(격자)
# plt.show()


###### 결과
# loss :  2682.103515625
# r2 :  0.4941245923104354
# rmse :  51.78902641766279

# learning_rate = 0.05
# 걸린시간 : 2.62 초
# loss :  3078.388427734375
# r2 :  0.4193806646076883
# rmse :  55.48322671848464