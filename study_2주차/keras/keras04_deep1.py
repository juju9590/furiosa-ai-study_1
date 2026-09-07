from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2. 모델구성 (딥러닝)
model = Sequential()
model.add(Dense(3, input_dim=1)) #1개 들어가서 3개 나옴
model.add(Dense(5, input_dim=3)) #3개 들어가서 5개 나옴
model.add(Dense(4, input_dim=5)) #5개 들어가서 4개 나옴
model.add(Dense(1, input_dim=4)) #4개 들어가서 1개 나옴

# 굳이 인풋데이터를 반복해서 쓸 필요 없음

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = 1500)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

result = model.predict(np.array([1,2,3,4,5]))
print("예측값 : ", result)

