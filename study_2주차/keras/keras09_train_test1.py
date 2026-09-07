# 통데이터로 했을경우 과적합 문제나 정확도가 낮아질 수 있다
# 그래서 데이터를 훈련과 테스트로 나눈다.
# 이유는 정확한 평가를 위해


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 트레인 vs 테스트 = 70 vs 30 분리

x_train =np.array([1,2,3,4,5,6,7])
y_train =np.array([1,2,3,4,5,6,7])

x_test =np.array([8,9,10])
y_test =np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=4)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
# 평가를 테스트 데이터로 하겠다. 훈련에 관여하지 않은 test값을 넣어야 한다 
# evaluate 에서는 배치사이즈는 적용 안된다
print('loss :', loss)

# [처음]
# model.add(Dense(1, input_dim=1))

# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 116.4130
# Epoch 99/100
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - loss: 116.1859
# Epoch 100/100
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - loss: 115.9639 <===== train 데이터의 loss
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - loss: 479.6761 <===== test 데이터의 loss
# loss : 479.6761169433594

# ================================================================================
# [두번째]
# model.add(Dense(3, input_dim=1))
# model.add(Dense(5))
# model.add(Dense(3))
# model.add(Dense(1))

# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - loss: 6.5174e-04
# Epoch 99/100
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - loss: 6.4474e-04
# Epoch 100/100
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - loss: 6.2538e-04
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.0022
# loss : 0.0022294053342193365

# ==> 결과를 보면 train loss와 test loss 중 어느것을 신뢰해야 하는가? 
# test loss 의 결과를 신뢰









