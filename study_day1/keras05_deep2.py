from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1)) # 레이어에 출력노드만 넣으면 됨
model.add(Dense(15)) 
model.add(Dense(8)) 
model.add(Dense(3)) 
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs = 1899)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

# result = model.predict(np.array([1,2,3,4,5,6]))
# print("예측값 : ", result)
# ctrl + /

