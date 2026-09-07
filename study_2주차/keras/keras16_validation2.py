from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train = x[:8]
y_train = y[:8]

x_val = x[8:12]
y_val = y[8:12]

x_test = x[12:]
y_test = y[12:]

print(x_train.shape, x_val.shape, x_test.shape)
print(x_train, x_val, x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=8))

#3. 컴파일, 훈련
model.compile(loss='mse', optimazer='adam')
model.fit(x_train, y_train, ecophs=100,
          verbose=1,
          validation_data=(x_val, y_val)
          )

#4. 평가, 성능
loss = model.evaluate(x_test, y_test)
print('loss :', loss)