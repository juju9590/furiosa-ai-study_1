import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10)]).T
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1],
              [9,8,7,6,5,4,3,2,1,0]]
              ).transpose()

print(x.shape, y.shape ) # (10, 1) (10, 3)
# 성능은 장담할 수 없지만 1개의 피처로 3개의 결과를 얻는 작업은 가능하다.

# [실습]
# 요구사항 : 11.00, 0.00, -1.00


#2. 모델구성
model = Sequential()
model.add(Dense(8, input_dim=1))
model.add(Dense(5))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=5)


#4. 평가, 예측
loss = model.evaluate(x,y)
print("loss :", loss)

results = model.predict(np.array([[10]])) # (1,1)
print("[10] 예측값 : ", results)

# loss : 1.3440182126955857e-11
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step
# [10] 예측값 :  [[ 1.1000002e+01  6.0200691e-06 -1.0000035e+00]]

# loss는 낮은데  예측값이 목표와 차이가 나거나, loss가 큰데 예측값이 목표값과 멀어질 경우 어느것을 골라야 할까?
# loss가 낮은 값을 선택해야 함

# 지금까지 모델링에서는 전체 데이터를 훈련 시킴, 추후 트레인(훈련):테스트(평가) 는  70:30으로 분리하여 훈련시킨 후 검증한다.

