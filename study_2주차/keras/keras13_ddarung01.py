# https://dacon.io/competitions/open/235576/overview/description

# 모델 훈련 시 필요없는 컬럼(인덱스) 제거
# LLM 에 들어가는 데이터
# LLM 커스터마이징 / PRE-TRAINED 하려면 이러한 내용을 알고 있어야 함

# test_csv는 y값(count)이 없음 -> 학습 불가
# train.csv로 훈련시키고(train/test) model.predict() 에 test_csv를 집어넣고 그 결과가 submission이 됨

# 결측치 처리 방법 많음 (10개~)
# 시간 단위 데이터는 1시간뒤의 시간을 중앙값으로도 가능한데
# 데이터 특징에 따라 처리 방법이 다름 처리했을 때 해당 값이 그럴 가능성이 높아야함
# 데이터 갯수가 많을 때 정확하지 않은 데이터는 삭제하는게 나음
# RMSLE가 평가지표여도 RMS(유사지표)로 훈련시켜서 잘 나온 것 넣기

# 로스가 계속 감소하면 에포크 늘려주기

import numpy as np # 수치 계산에 특화
import pandas as pd # sklearn 만큼 강력함
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
# path = "./_data/ddarung/" # 상대경로
# path = "c:/study/_data/ddarung/" # 절대경로
path = "c:\study\_data\ddarung/" # 슬래시 역슬래시 / 2개 상관없음, 섞어쓰기 되지만 가급적 비권장
# path = "c://study//_data//ddarung/"
# path = "c:\\study\\_data\\ddarung/"

# train_csv는 x,y(count) 가 합쳐진 상태로, 분리해줘야함
train_csv = pd.read_csv(path + "train.csv", index_col=0) # csv 가져오기, 인덱스 컬럼 데이터에 포함 x 
print(train_csv) # id열 포함 [1459 rows x 11 columns] -> id열 안 포함 [1459 rows x 10 columns] index_col=0 때문

test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv) # [715 rows x 9 columns]

submission = pd.read_csv(path + "submission.csv", index_col=0)
print(submission) # [715 rows x 1 columns]

# 데이터 받으면 shape 찍기
print(train_csv.shape) # (1459, 10) // 훈련에서 분리해야 하는 데이터 train_test_split
print(test_csv.shape) # (715, 9) // 제출용 파일, y 없음
print(submission.shape) # (715, 1)

print(train_csv.columns) # 컬럼이 중요함
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='str')

print(train_csv.info())
print(test_csv.info())

# exit()
############################# 결측치 처리 1. 삭제 ################################
# 데이터가 많을 때는 삭제 고려 가능
# 모델 성능이 좋으려면 데이터를 최대한 살려야함
train_csv = train_csv.dropna() # 결측치 있는 row 자체를 없애버림
print(train_csv) # [1328 rows x 10 columns]

# train_csv를 x와 y로 분리
x = train_csv.drop(['count'], axis=1) # 열(컬럼) 삭제 | 행0, 열1
print(x) # [1328 rows x 9 columns]

y = train_csv['count'] # pandas에서 컬럼만 빼는거
print(y)
print(y.shape) # (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=141)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=9))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=250, batch_size=16)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
print("r2 : ", r2_score(y_test, y_predict))
print("rmse : ", np.sqrt(mean_squared_error(y_test, y_predict)))


# train_size=0.9, random_state=141
# model = Sequential()
# model.add(Dense(5, input_dim=9))
# model.add(Dense(1))
# 75/75 ━━━━━━━━━━━━━━━━━━━━ 0s 647us/step - loss: 2933.3904
# loss :  2460.31005859375
# r2 :  0.6233004671069471
# rmse :  47.60151403059909

"""
### 하이퍼파라미터 튜닝 ###
random_state
train_size
레이어 깊이
노드 갯수
epochs
batch_size
"""

# 명세 정리하기
'''
1차 시도
random : 337
train_size : 0.75
epochs = 50
batch_size = 1
model.add(Dense(5, input_dim=9))
model.add(Dense(1))

결과
rmse : 37
r2 : 0.66
'''

'''
2차 시도
random : 337
train_size : 0.75
epochs = 200
batch_size = 32
model.add(Dense(5, input_dim=9))
model.add(Dense(1))

결과
rmse : 27
r2 : 0.67
'''