from sklearn.datasets import fetch_california_housing, load_diabetes
# 사이킷런에서 학습 데이터 제공 datasets (캘리포이아 집값), 데이터를 함수 형태로 만들어 놨음
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    test_size=0.3,
    random_state=389,
    shuffle=True
)

#2. 모델
model = Sequential()
model.add(Dense(4, input_dim=10))
model.add(Dense(6))
model.add(Dense(24))
model.add(Dense(64))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=24)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print(loss)

# Epoch 100/100
# 29/29 ━━━━━━━━━━━━━━━━━━━━ 0s 964us/step - loss: 2937.8433
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 2885.7354 
# 2885.7353515625

# Epoch 100/100
# 309/309 ━━━━━━━━━━━━━━━━━━━━ 0s 628us/step - loss: 3007.4512
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step - loss: 2825.8079  
# 2825.807861328125

# Epoch 100/100
# 78/78 ━━━━━━━━━━━━━━━━━━━━ 0s 854us/step - loss: 2995.3484
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 2809.8445 
# 2809.844482421875

# 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 3010.9316 
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 2732.0415 
# 2732.04150390625

# Epoch 100/100
# 9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 3312.1040 
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 2725.1587 
# 2725.15869140625

# Epoch 155/155
# 9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 3168.6184 
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 2654.1597 
# 2654.15966796875

# 9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 3093.8643 
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 2621.2781 
# 2621.278076171875

# Epoch 172/172
# 9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 3109.4351 
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 2598.9204 
# 2598.92041015625