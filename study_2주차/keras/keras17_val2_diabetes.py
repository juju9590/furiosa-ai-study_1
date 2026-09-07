# 12-3 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=777)

# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=500, batch_size=48,
          validation_split=0.33,
          )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)

# 23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 2820.7498 
# loss :  3197.64501953125
# r2 :  0.4952925921680489

# Epoch 499/500
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 2925.2998 - val_loss: 3032.3916
# Epoch 500/500
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 2925.0405 - val_loss: 3027.1514
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 3255.6702 
# loss :  3255.670166015625
# 5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
# r2 :  0.47296724228506015

