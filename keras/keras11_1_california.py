# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# ㄴ 사이킷런에서 데이터셋이 다운되지 않을 경우 위 내용으로 처리하면 된다.




from sklearn.datasets import fetch_california_housing
# 사이킷런에서 학습 데이터 제공 datasets (캘리포이아 집값), 데이터를 함수 형태로 만들어 놨음
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
datasets = fetch_california_housing() # 켈리포이나아 하우징 데이터를 앞으로 데이터 셋이라고 할꺼야
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(20640, 8) (20640,) => 8열 =컬럼 = 피처

x_train, x_test, y_train, y_test = train_test_split(
# 데이터의 순서는 바뀌면 안된다.
    x,y,
    train_size=0.85,
    test_size=0.15,
    shuffle=True,
    random_state=567,
)

#2. 모델구성
model = Sequential()
model.add(Dense(2, input_dim=8))
model.add(Dense(6))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=80,) #batch_size=150

print("=================================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, ) #여기도 배치사이즈가 디폴트값으로 들어가 있다
print("loss :", loss)

# Epoch 50/50
# 331/331 ━━━━━━━━━━━━━━━━━━━━ 0s 666us/step - loss: 1.4585
# =================================================
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 864us/step - loss: 0.7512

# Epoch 80/80
# 516/516 ━━━━━━━━━━━━━━━━━━━━ 0s 667us/step - loss: 0.7641
# =================================================
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 666us/step - loss: 0.6717
# loss : 0.6717491149902344

# Epoch 80/80
# 516/516 ━━━━━━━━━━━━━━━━━━━━ 0s 625us/step - loss: 0.6441
# =================================================
# 129/129 ━━━━━━━━━━━━━━━━━━━━ 0s 769us/step - loss: 0.6355
# loss : 0.635458767414093

# Epoch 80/80
# 549/549 ━━━━━━━━━━━━━━━━━━━━ 0s 708us/step - loss: 0.6333
# =================================================
# 97/97 ━━━━━━━━━━━━━━━━━━━━ 0s 733us/step - loss: 0.6161
# loss : 0.6160901188850403

# predict는 아직 하지 말것


