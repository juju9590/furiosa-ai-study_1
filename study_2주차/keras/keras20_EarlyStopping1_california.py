# 19-1 copy

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
datasets = fetch_california_housing() # 켈리포이나아 하우징 데이터를 앞으로 데이터 셋이라고 할꺼야
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(20640, 8) (20640,) => 8열 =컬럼 = 피처

x_train, x_test, y_train, y_test = train_test_split(
# 데이터의 순서는 바뀌면 안된다.
    x,y,
    train_size=0.8,
    test_size=0.2,
    shuffle=True,
    random_state=333,
)

#2. 모델구성
model = Sequential()
model.add(Dense(6, input_dim=8))
model.add(Dense(2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping #이름:카멜케이스/클래스

es = EarlyStopping(
    monitor='val_loss', #loss도 가능
    mode='min', # loss는 min, auto 로 잡아준다
    patience=15, # 갱신이 안되면 10번 움직임
    restore_best_weights=True, #최소값의 지점의 가중치로 적용(원칙은 True)
)
# 여기서는 EarlyStopping 기준을 정하고, Fit에 넣어줘야 학습에 적용된다.
#  loss = min, acculucy = max, auto로하면 로스는 민, 엑큐러스는 맥스로 자동설정

start_time = time.time() #현재시간을 반환, = 시작시간
# [기능확인]
hist = model.fit(x_train, y_train, 
                epochs=500000, 
                batch_size=64,
                validation_split=0.2,
                callbacks=[es],
                )
# fit() 함수였고 기본적으로 출력/반복 기능이 있다.
# EarlyStopping
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
plt.plot(hist.history['loss'][2:], c='red', label='loss')
plt.plot(hist.history['val_loss'][2:], c='blue', label='val_loss')
# x를 명시하지 않으면 y값을 시간순으로 그려줌
plt.legend(loc="upper right") # 우측 상단에 라벨표시(범례)
plt.title('California 캘리포니아 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid() #모눈종이처럼 표시(격자)
plt.show()