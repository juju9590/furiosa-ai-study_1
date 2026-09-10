# 19-1 copy

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_california_housing
# 사이킷런에서 학습 데이터 제공 datasets (캘리포이아 집값), 데이터를 함수 형태로 만들어 놨음
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

print(x.shape, y.shape) #(20640, 8) (20640,)

'''
MinMaxScaler 알고리즘
ㄴ X 데이터를 (0~1)사이로 만들어 준다 (x 데이터를 동일한 비율로 조절한다)

       원값 - Min
수식 = -----------
       Max - Min
'''

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x) # 실행 준비
x = scaler.transform(x) # 실행 준비한걸 변환 시킨다

print(x)
'''
[[0.53966842 0.78431373 0.0435123  ... 0.00149943 0.5674814  0.21115538]
 [0.53802706 0.39215686 0.03822395 ... 0.00114074 0.565356   0.21215139]
 [0.46602805 1.         0.05275646 ... 0.00169796 0.5642933  0.21015936]
 ...
 [0.08276438 0.31372549 0.03090386 ... 0.0013144  0.73219979 0.31175299]
 [0.09429525 0.33333333 0.03178269 ... 0.0011515  0.73219979 0.30179283]
 [0.13025338 0.29411765 0.03125246 ... 0.00154886 0.72582359 0.30976096]]
'''
print(np.min(x), np.max(x)) # 0.0 1.0000000000000002

# exit()


x_train, x_test, y_train, y_test = train_test_split(
# 데이터의 순서는 바뀌면 안된다.
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
model.compile(loss='mse', optimizer='adam')

start_time = time.time() #현재시간을 반환, = 시작시간
# [기능확인]
hist = model.fit(x_train, y_train, 
                epochs=100, batch_size=32,
                validation_split=0.2
                ) #batch_size=150
# fit() 함수였고 기본적으로 출력/반복 기능이 있다.
end_time = time.time() # 현재시간을 반환, = 끝시간

print("=================================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, ) #여기도 배치사이즈가 디폴트값으로 들어가 있다
print("loss :", loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 : ", round(r2,3))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", round(rmse,3))

# print("걸린시간 :", round(end_time - start_time,2), "초")/

print("================== history ===============================")
print(hist)
print("================== hist.history ==========================")
print(hist.history)
# mode.fit()에서 loss와 val_loss를 출력하고 있음을 알 수있다.
# 딕셔너리 = 키 : 벨류 { }
# 2개 이상은 리스트 [ ]
print("================== loss ==========================")
print(hist.history['loss'])
print("================== val_loss ==========================")
print(hist.history['val_loss'])

print("================== 시각화 ==========================")
import matplotlib.pyplot as plt

# plt에서 한글 못 읽기 때문에 맑은고딕 폰트 설정 필수
# plt.rcParams['font.family']='Malgun Gothic'
plt.rc('font', family = 'Hancom Gothic')

plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# x를 명시하지 않으면 y값을 시간순으로 그려줌
plt.plot(hist.history['loss'][2:], c='red', label='loss')
plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss')
# x를 명시하지 않으면 y값을 시간순으로 그려줌
plt.legend(loc="upper right") # 우측 상단에 라벨표시(범례)
plt.title('California 캘리포니아 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid() #모눈종이처럼 표시(격자)
plt.show()


# Epoch 100/100
# 413/413 ━━━━━━━━━━━━━━━━━━━━ 0s 852us/step - loss: 1.3361 - val_loss: 1.3667
# =================================================
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 620us/step - loss: 1.2891
# loss : 1.2890541553497314
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 399us/step
# r2 :  -0.0
# rmse :  1.135





