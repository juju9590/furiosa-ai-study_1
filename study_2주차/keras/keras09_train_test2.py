#09_1 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape, y.shape)

# 트레인 vs 테스트 = 70 vs 30 분리

# x_train =np.array([1,2,3,4,5,6,7])
# y_train =np.array([1,2,3,4,5,6,7])

# x_test =np.array([8,9,10])
# y_test =np.array([8,9,10])

#[찾아보기] 넘파이 리스트의 슬라이싱 => 7:3 으로 나누기

x_train = x[:7] # x[0:7]
y_train = y[:7] # y[0:7]

x_test = x[7:] # x[7:10]
y_test = y[7:] # y[7:10]

print(x_train.shape, y_train.shape ,x_train, y_train)
print(x_test.shape, y_test.shape, x_test, y_test)

# [1 2 3 4 5 6 7] [1 2 3 4 5 6 7]
# [ 8  9 10] [ 8  9 10]

# (10,) (10,)
# (7,) (7,) [1 2 3 4 5 6 7] [1 2 3 4 5 6 7]
# (3,) (3,) [ 8  9 10] [ 8  9 10]

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

# Epoch 98/100
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 0.0160
# Epoch 99/100
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 0.0159
# Epoch 100/100
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - loss: 0.0159
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step - loss: 0.0593
# loss : 0.05927187204360962

# 현재 10개의 데이터 중에서 순차적으로 0~7, 8~10 으로 데이터 셋을 분리 하였지만
# 랜덤으로 70 vs 30 분리하여야 신뢰도가 높은 결과를 구할 수 있다.


