import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10)) # [0 1 2 3 4 5 6 7 8 9]
# range(10) = 0 ~ 9 (10-1)
print(x) 

x = np.array(range(1, 10)) # [1 2 3 4 5 6 7 8 9]
print(x) 

x = np.array(range(1, 11)) # [ 1  2  3  4  5  6  7  8  9 10]
print(x) 

x = np.array([range(10),range(21,31,),range(201,211)]).T
# 에러의 라인 찾기, 에러의 이름
print(x.shape) # (3, 10) --> (10, 3) (x값 전치)

y = np.array(range(1,11))
print(y.shape) # (10,)

# [실습]
# [10, 31, 211] , 11.00 뜨면 합격

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=3))
model.add(Dense(5))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=6)

#4. 평가, 예측
loss = model.evaluate(x,y)
print("loss :", loss)

results = model.predict(np.array([[10, 31, 211]])) # (1,3)
print("[10, 31, 211] 예측값 : ", results)

# loss : 5.6644467762156925e-12
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 42ms/step
# [10, 31, 211] 예측값 :  [[11.000008]]



