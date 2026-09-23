# 러닝레이트
# 27-1 copy

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import time


#1. 데이터
datasets = fetch_california_housing() 
x = datasets.data
y = datasets.target

print(x.shape, y.shape) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x) 
x = scaler.transform(x) 


print(np.min(x), np.max(x)) 

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.8,
    # test_size=0.2,
    # shuffle=True, #디폴트 섞는다
    random_state=777,
) 

#2. 모델구성
model = Sequential()
model.add(Dense(2, activation='relu', input_dim=8))
model.add(Dense(6, activation='relu'))
model.add(Dense(12))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
# learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
# learning_rate = 0.005
learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=50,
    verbose=1,
    restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5, # learning_rate(러닝레이트) 비율 조절

)
# 아담에는 자동조절 장치가 있다.. 
# 하지만 Reduce와 같이 쓰면 보완되어 성능이 더 좋을 수 있다.
# Reduce 줄이다

start_time = time.time() 

hist = model.fit(x_train, y_train, 
                epochs=300, 
                batch_size=32,
                validation_split=0.2,
                callbacks =[es, rlr,],
                verbose=1,
                ) 

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, ) 
print("loss :", loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 : ", round(r2,3))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", round(rmse,3))

print("걸린시간 :", round(end_time - start_time,2), "초")

# print("================== history ===============================")
# print(hist)
# print("================== hist.history ==========================")
# print(hist.history)
# # mode.fit()에서 loss와 val_loss를 출력하고 있음을 알 수있다.
# # 딕셔너리 = 키 : 벨류 { }
# # 2개 이상은 리스트 [ ]
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
# # plt.plot(hist.history['loss'], c='red', label='loss')
# # plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# # x를 명시하지 않으면 y값을 시간순으로 그려줌
# plt.plot(hist.history['loss'][2:], c='red', label='loss')
# plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss')
# # x를 명시하지 않으면 y값을 시간순으로 그려줌
# plt.legend(loc="upper right") # 우측 상단에 라벨표시(범례)
# plt.title('California 캘리포니아 Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid() #모눈종이처럼 표시(격자)
# plt.show()

###### 결과
# loss : 1.2890541553497314
# r2 :  -0.0
# rmse :  1.135

##### learning_rate = 0.01
# loss : 0.3875690698623657
# r2 :  0.699
# rmse :  0.623
# 걸린시간 : 268.19 초

##### learning_rate = 0.01
# Epoch 110: early stopping
# loss : 1.2890976667404175
# r2 :  -0.0
# rmse :  1.135
# 걸린시간 : 74.65 초




