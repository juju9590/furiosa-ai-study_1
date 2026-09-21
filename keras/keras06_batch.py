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


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #최대한 디폴트 옵션
model.fit(x, y, epochs = 12411, batch_size=1)
# 대규모 데이터는 잘라서 작업 = 훈련할때 배치사이즈로 작업
# 왜 배치사이즈를 사용하는가? 
# 빅데이터의 경우 훈련시 문제 발생
# 같은 데이터를 가지고 더 많은 훈련을 시킬 수 있다
# 나중에는 랜덤하게 배치를 해서 훈련할 수 있다 
# 배치사이즈 기본(디폴트)는 32이고, 디폴트로 했을때 85점정도 나옴


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

# 텐서(Tensor)는 스칼라, 벡터, 행렬을 포함하여 0차원부터 N차원까지 확장된 다차원 숫자 배열의 집합
# 1, 2, 3 ... 각각을 스칼라라고함 (데이터 1개) == 0차원
# [1,2,3] 묶어 놓으면  벡터라고함  == 1차원 
# [ [1,2,3], [1,2,3], [1,2,3]]  벡터를 묶어 놓으면 행렬 = 메트릭스 == 2차원
# 행렬이 여러개 묶여 있으면 텐서 == 3차원
# 탠서가 모여있으면 4차원 텐서, 5차원 텐서, 6차원 텐서...

# 예시, 엑셀 가로세로 있으면 2차원, 이미지는 4차원 데이터

# 개발할때 가장 많이 틀리는경우
# 1. 오타
# 2. 쉐이프
# 3. 버전
# 4. 경로



