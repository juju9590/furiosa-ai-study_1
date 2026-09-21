import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21,31), range(201, 211)]).T
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1]]
              ).transpose()

print(x.shape, y.shape ) #(3, 10) (2, 10) --> (10, 3) (10, 2)
# input_dim=3,  Outputlayer=2

# print(x)
# print(y)

# [실습]
# [10, 31, 211]의 예측값
#  요구사항 11.00, 0.00

#2. 모델구성
model = Sequential()
model.add(Dense(8, input_dim=3))
model.add(Dense(5))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=3)


#4. 평가, 예측
loss = model.evaluate(x,y)
print("loss :", loss)

results = model.predict(np.array([[10, 31, 211]])) # (1,3)
print("[10, 31, 211] 예측값 : ", results)

# loss : 3.4340602583782243e-10
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step
# [10, 31, 211] 예측값 :  [[1.1000007e+01 3.0668569e-05]]



