from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1)) #1개 들어가서 3개 나옴
model.add(Dense(4)) 
model.add(Dense(5)) #3개 들어가서 5개 나옴
model.add(Dense(4)) #5개 들어가서 4개 나옴
model.add(Dense(3))  
model.add(Dense(1)) #4개 들어가서 1개 나옴
# 하이퍼파라미터 튜닝
# 1. 레이어 노드/ 층 
# 2. 훈련할때 배치사이즈 조정

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #최대한 디폴트 옵션
model.fit(x, y, epochs = 12411, batch_size=1)
# 대규모 데이터는 잘라서 작업 = 훈련할때 배치사이즈로 작업


#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

# result = model.predict(np.array([1,2,3,4,5]))
# print("예측값 : ", result)

# 성능향상을 위해 할 수 있는 방법 
# 1. 에포크 조정
# 2. 노드의 수
# 3. 레이어 깊이 
# 4. 배치 사이즈

