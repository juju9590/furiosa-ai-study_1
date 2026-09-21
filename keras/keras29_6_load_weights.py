# 29-3 카피

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_california_housing
# 사이킷런에서 학습 데이터 제공 datasets (캘리포이아 집값), 데이터를 함수 형태로 만들어 놨음
from tensorflow.keras.models import Sequential, load_model
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

print(x.shape, y.shape) #(20640, 8) (20640,)


x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.8,
    # test_size=0.2,
    # shuffle=True, #디폴트 섞는다
    random_state=777,
) 

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler() # 이상치에 강력

# x_train = scaler.transform(x_train)  # <===  x_train 스케일링 시행
# x_test = scaler.transform(x_test)    # <===  x_test 스케일링 시행

x_train = scaler.fit_transform(x_train) # 한줄로 사욯 가능
x_test = scaler.transform(x_test)    


print(np.min(x_train), np.max(x_train)) 
print(np.min(x_test), np.max(x_test)) 

#2. 모델구성
model = Sequential()
model.add(Dense(2, activation='relu', input_dim=8))
model.add(Dense(6, activation='relu'))
model.add(Dense(12))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

# model.summary()

path ='./_save/keras29/' # 저장 경로
# # model.save(path + 'keras29_1_save_model.keras') # 초기 가중치 + 모델 저장
# model.save_weights(path + 'keras29_5_save_1.weights.h5') # 확장자 .weights.h5/ 훈련되지 않은 웨이트 저장 (초기 가중치)
# ㄴ save_weights는 순수 가중치만 저장이 되기 때문에 모델구성을 살려 두어야 한다.


# # model = load_model(path + 'keras29_1_save_model.keras')
# model.load_weights(path + 'keras29_5_save_1.weights.h5')  
model.load_weights(path + 'keras29_5_save_2.weights.h5')


model.summary() # 모델구조 보기

# exit()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # 순수 가중치만 불러오기 때문에 컴파일 살려야 함  

# import time
# start_time = time.time()

# hist = model.fit(x_train, y_train, 
#                 epochs=100, batch_size=32,
#                 validation_split=0.2
#                 ) #batch_size=150
# # fit() 함수였고 기본적으로 출력/반복 기능이 있다.

# end_time = time.time() # 현재시간을 반환, = 끝시간
# print("걸린시간 :", round(end_time-start_time,2), "초")


# model.save(path + 'keras29_3_save_model.keras') # 가중치 갱신이 있는 상태 (모델 + 훈련) 저장
# model.save_weights(path + 'keras29_5_save_2.weights.h5') # 확장자 .weights.h5 / 100번째의 가중치 저장된 상태






print("=================== 학습 종료 ========================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, )
print("loss :", loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 : ", round(r2,3))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", round(rmse,3))

# print("걸린시간 :", round(end_time - start_time,2), "초")/

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

######################## 결과 ##################################################

# Epoch 100/100
# 413/413 ━━━━━━━━━━━━━━━━━━━━ 0s 807us/step - loss: 0.4926 - val_loss: 0.5594
# =================================================
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 579us/step - loss: 0.5388
# loss : 0.538833737373352
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 472us/step
# r2 :  0.577
# rmse :  0.734


######################## train_test 분리 전 MinMaxScaler 적용 결과 ################## ==> 하향
# Epoch 100/100
# 413/413 ━━━━━━━━━━━━━━━━━━━━ 0s 852us/step - loss: 1.3361 - val_loss: 1.3667
# =================================================
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 620us/step - loss: 1.2891
# loss : 1.2890541553497314
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 399us/step
# r2 :  -0.0
# rmse :  1.135

# ####################### StandardScaler 적용 결과 ############## ==> 향상
# Epoch 100/100
# 413/413 ━━━━━━━━━━━━━━━━━━━━ 0s 834us/step - loss: 0.3711 - val_loss: 0.4006
# 걸린시간 : 37.61 초
# =================== 학습 종료 ========================
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 607us/step - loss: 0.3609
# loss : 0.3609026074409485
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 481us/step
# r2 :  0.72
# rmse :  0.601

# ####################### MaxAbsScaler 적용 결과 ############## ==> 성능하향
# Epoch 100/100
# 413/413 ━━━━━━━━━━━━━━━━━━━━ 0s 889us/step - loss: 0.4877 - val_loss: 0.4929
# 걸린시간 : 38.16 초
# =================== 학습 종료 ========================
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 606us/step - loss: 0.4673
# loss : 0.4672805368900299
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 422us/step
# r2 :  0.637
# rmse :  0.684


# ####################### RobustScaler 적용 결과 ############## ==> 성능향상
# Epoch 100/100
# 413/413 ━━━━━━━━━━━━━━━━━━━━ 0s 839us/step - loss: 0.3674 - val_loss: 0.3792
# 걸린시간 : 38.12 초
# =================== 학습 종료 ========================
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 588us/step - loss: 0.3700
# loss : 0.3699517250061035
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 331us/step
# r2 :  0.713
# rmse :  0.608
# 이상치에 강력한 놈 






