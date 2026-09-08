# 14-1 카피


# https://www.kaggle.com/competitions/bike-sharing-demand/overview

# 활성 함수
# Linear
# relu : 0이상은 그대로, 음수는 0처리

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping #학습 중 조기학습 종료 시, = 최소값을 구하는 알고리즘

es = EarlyStopping(monitor='val_loss', #loss 보다 조금 더 신뢰성 있는 val_loss로 선택. loss도 가능
              mode='min', # loss는 min, accuracy는 max, 잘 모르면 auto
              patience=20, # 가장 최소점에서 다음 최소점을 찾기위해 시도하는 횟수
              restore_best_weights=True, # 가장 최소점을 찾고나서 다음 20번 시도시 최소점이 없을때 가장 작은 최소값을 선택
              )

# 1. 데이터
path = "./_data/kaggle_bike/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# 데이터 확인
print(train_csv) # [10886 rows x 11 columns]
print(test_csv) # [6493 rows x 8 columns]
print(submission) # [6493 rows x 1 columns]

# shape 확인
print(train_csv.shape) # (10886, 11)
print(test_csv.shape) # (6493, 8)
print(submission.shape) # (6493, 1)

# 결측치 확인
print(train_csv.info()) # 결측치 X
print(test_csv.info()) # 결측치 X

print(train_csv.describe()) # 이상치 확인 가능 (봄~겨울 1~4)

print(train_csv.isna().sum()) # 컬럼별 결측치 확인
print(test_csv.isnull().sum()) # 컬럼별 결측치 확인

########### x, y 분리 #############3

x = train_csv.drop(['casual', 'registered', 'count'], axis=1) #열(컬럼) 삭제 
print(x) # [10886 rows x 8 columns]

y = train_csv['count']
print(y) # (10886,)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=142,
    )

# 2. 모델 구성
model = Sequential()
model.add(Dense(12, activation='relu', input_dim=8))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
hist = model.fit(x_train, y_train, 
                 epochs=100, batch_size=32,
                 validation_split=0.2,
                 callbacks=[es],
          )

print("=================================================")

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", rmse)


print("================== history ===============================")
print(hist)
print("================== hist.history ==========================")
print(hist.history)
print("================== loss ==========================")
print(hist.history['loss'])
print("================== val_loss ==========================")
print(hist.history['val_loss'])

print("================== 시각화 ==========================")
import matplotlib.pyplot as plt

# plt에서 한글 못 읽기 때문에 맑은고딕 폰트 설정 필수
# plt.rcParams['font.family']='Malgun Gothic' #한글폰트변경 1줄
plt.rc('font', family = 'Hancom Gothic') #한글폰트변경  1줄

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# x를 명시하지 않으면 y값을 시간순으로 그려줌
plt.legend(loc="upper right") # 우측 상단에 라벨표시(범례)
plt.title('캐글 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid() #모눈종이처럼 표시(격자)
plt.show()

# 최소의 오차(loss), 최적의 웨이트
# 그래프가 우하향 시 학습 증가
# 그래프가 과적합 현상 시 하이퍼 파라미터 튜닝 필요
# ㄴ 과적합이 걸리면 그래프에서 우하향하다가 왔다갔다 핑퐁을 치면 그 부분을 과적합으로 보고 하이퍼 파라미터 튜닝한다.
# 하이퍼파라미터란 내가 수정할 수 있는 항목 : epochs, batch_size, dence, EarlyStoppin 등등등
# 학습 시작시 초기 웨이트는 랜덤값,
# 얼리스탑핑을 할때는 에포를 크게 주고 patience로 제어를 한다.
# ㄴ restore_best_weights=True로 하면 최소값 선정, 
# ㄴ restore_best_weights=False는 최소값에서 patience 만큼 움직인 값
# ㄴ 2가지 다 해봐야한다 

# 다음시간은 "다중분류", "이진분류"
# local minima, global minima(파랑새)
# loss는 0 이상! 왜? 플러스 마이너스로 상계처리되면 안되니깐. (대부분 양수로 보이도록 조치가 되어있음)
# 분류는 딱딱 떨어지는거, 예시) 남자/여자, 라벨이 2개면 이진분류, 3개이상이면 다중분류...
# ㄴ 조건 : 라벨의 종류가 정해져 있음


