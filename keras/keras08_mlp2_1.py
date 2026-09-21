import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
              [9,8,7,6,5,1,3,2,1,0]
              ])
x = x.T
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape) #(3, 10) ==> x.T 적용 후 (10, 3)
print(y.shape) #(10,)

# 모델구성
model = Sequential()
model.add(Dense(5,input_dim=3))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1001, batch_size=4)

# 평가, 예측
loss = model.evaluate(x,y)
print("loss : ", loss)

results = model.predict(np.array([[10, 1.3, 0]])) # (1,3) 행무시 열우선
print("[10, 1.3, 0]의 예측값 : ",  results)

# # 0.00015 이하면 
# # [10, 1.3, 0] 예측값 10.00 정도 나오면

# loss :  1.5311077277146978e-06
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step
# [10, 1.3, 0]의 예측값 :  [[10.000121]]
# PS C:\study> 
