import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  # ctrl + space
import numpy as np


#1. 데이터 : 데이터 받으면 제일 먼저 할 일 ==>  Shape(쉐이프) 확인하기
x = np.array([[1,2,3,4,5],
              [6,7,8,9,10]]) # x = (2, 5)
x = x.T
# x = x.transpose() # 함수

# [네이밍 룰]
# 스네이크 케이스 : keras_deep_aaa.py
# 카멜 케이스 : kerasDeepAaa.py

# x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]]) 
y = np.array([1,2,3,4,5]) 

print(x.shape) # (5, 2)
print(y.shape) # (5,)

# X, y의 행의 갯수가 같은지 확인하고, 모델구성에서 input_dim= 열로 넣으면 된다

#2. 모델구성
model = Sequential() 
# 대문자 S로 시작해서 ()가 있음 클래스 = 통상적, 100%는 아님
# 소문자로 시작해서 ()가 있음 함수 = 기능이 있는 것
# ㄴ 가급적 위와 같이 지정해서 사용 (약속)

model.add(Dense(5, input_dim=2))
# print(x.shape) # (5, 2) 5행 2열 => 행 무시 열 우선
# 모델구성할때 input_dim= 열의 갯수이다 => (5,2) 데이터에서 열의 갯수는 2니깐 input_dim=2 <=== 행 무시 열 우선 
# 컬럼이 많으면 많을수록 성능이 좋아진다. (단, 상관관계가 높아야 하겠지)
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=3)

#4. 평가, 예측 
loss = model.evaluate(x, y)
print("loss :", loss)

results = model.predict(np.array([[6,11]])) # (1, 2) <== 행 무시 열 우선
print("[6, 11]예측값 : ", results)

