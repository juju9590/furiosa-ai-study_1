import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,7,9,3,8,12,13, 8,14,15,16, 9, 6,17,23,20])

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    test_size=0.3,
    shuffle=True,
    random_state=888,
    )

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=4)

print("================================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss :", loss)

results = model.predict(x)
print("예측값 :", results)


# 시각화
plt.scatter(x,y, color='gray')
plt.plot(x,results)
plt.show()

# Epoch 100/100
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 8.7185  
# ================================================
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step - loss: 20.6986
# loss : 20.69858741760254
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step
# 예측값 : [[ 1.0420675]
#  [ 2.03997  ]
#  [ 3.037872 ]
#  [ 4.0357723]
#  [ 5.0336766]
#  [ 6.0315776]
#  [ 7.029481 ]
#  [ 8.027384 ]
#  [ 9.025286 ]
#  [10.023187 ]
#  [11.021089 ]
#  [12.018991 ]
#  [13.016891 ]
#  [14.014795 ]
#  [15.012696 ]
#  [16.010595 ]
#  [17.0085   ]
#  [18.006403 ]
#  [19.004307 ]
#  [20.002213 ]]


