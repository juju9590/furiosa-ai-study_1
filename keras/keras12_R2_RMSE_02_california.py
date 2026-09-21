# r^2 0.55 이상

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=777)

# 2. 모델
model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(4, input_dim=8))
model.add(Dense(1, input_dim=8))

# 3. 컴파일, 학습
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=1000, batch_size=32)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test) # loss
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)

# 194/194 ━━━━━━━━━━━━━━━━━━━━ 0s 627us/step - loss: 0.6266
# loss :  0.6265900135040283
# r2 :  0.532679893338787