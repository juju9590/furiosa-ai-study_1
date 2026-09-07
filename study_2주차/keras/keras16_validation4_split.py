# 16-3 카피

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))


x_train, x_test, y_train, y_test = train_test_split(
    x, y,train_size=0.75, random_state= 42, )


print(x_train)
print(x_test)
# print(x_val)

# 2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=4,
          verbose=1,
          validation_split=0.33,
          )

# validation_split = 훈련중에 검증을 하겠다!!!

print("==================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)