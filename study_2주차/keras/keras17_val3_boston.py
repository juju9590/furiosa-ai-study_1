#13-보스턴 copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing

# 텐서플로우에서 데이터셋을 가져올때 아래와 같이 가져오면 된다

(x_train, y_train), (x_test, y_test)= boston_housing.load_data()
print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
print(y_train.shape, y_test.shape) #(404,) (102,)

#2. 모델구성
model = Sequential()
model.add(Dense(5,input_dim=13))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
# mse 계산 = (y트레인 - y트레인 예측값)의 제곱을 합해서 총 데이터 갯수로 나눠준다 (식 확인)
# 파이썬은 인터프리터언어로 epochs 할때마다 loss 값을 계산하여 W(가중치)를 갱신한다
model.fit(x_train, y_train, epochs=100, batch_size=4,
          validation_split=0.33,
          )

# 훈련이 종료되면 마지막 W값이 정해진다.

print("=====================================")

#4. 평가, 성능
loss = model.evaluate(x_test, y_test, )
# 최종 정해진 W값으로 순전파 y=wx+b를 계산하여 loss값을 계산한다. 
# 단, 여기서 계산되는 y의 원값은 y_test, 최종 W를 적용하여 y_test 예측값 구함
# y_test, y_test 예측값 으로 loss 계산

print(loss)

# results = model.predict(x) 
# 여기서 x는 x_test를 넣어서 y=wx+b로 계산하여 results 를 추출한다.

# Epoch 99/100
# 68/68 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 34.3188 - val_loss: 54.2071
# Epoch 100/100
# 68/68 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 35.8418 - val_loss: 44.7281
# =====================================
# 4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 38.5271 
# 38.527095794677734
