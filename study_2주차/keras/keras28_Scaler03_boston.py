#13-보스턴 copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np


# 텐서플로우에서 데이터셋을 가져올때 아래와 같이 가져오면 된다

(x_train, y_train), (x_test, y_test)= boston_housing.load_data()
print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
print(y_train.shape, y_test.shape) #(404,) (102,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

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

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# mse 계산 = (y트레인 - y트레인 예측값)의 제곱을 합해서 총 데이터 갯수로 나눠준다 (식 확인)
# 파이썬은 인터프리터언어로 epochs 할때마다 loss 값을 계산하여 W(가중치)를 갱신한다
hist = model.fit(x_train, y_train, 
          epochs=200, batch_size=64,
          validation_split=0.2,
          )

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
# plt.title('보스턴 Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid() #모눈종이처럼 표시(격자)
# plt.show()

# 그래프가 우하향 하고 있다면 훈련 더 시키기
# 그래프가 핑퐁하고 있다면 하이퍼파라미터 튜닝

################################## 결과 ###################################
# Epoch 200/200
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 54.7659 - val_loss: 67.8529
# =====================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 67.0453 
# 67.04534149169922
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
# r2 :  0.19459107175796264
# rmse :  8.188121812667095

############################ MinMaxScaler 후 ########################   향상
# Epoch 200/200
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 39.4716 - val_loss: 41.4484
# =====================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 39.9959 
# 39.99590301513672
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
# r2 :  0.5195332540051056
# rmse :  6.324231259402225