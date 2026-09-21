from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# 최소의 loss로 최적의 웨이트를 구한다
model.fit(x, y, epochs=999)

#4. 평가, 예측
loss = model.evaluate(x, y) #평가
print("loss :", loss)

result = model.predict(np.array([1,2,3,4,5,6,7])) #예측
print("7의 예측값 :", result)

# 데이터가 조금 더 많을 경우 조금 더 적은 에포크로 최적값을 가져올 수 있다.








