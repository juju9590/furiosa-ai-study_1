import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [검색] train과 test를 섞어서 7:3 나눈다
# 힌트 : 사이킷런

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    # train_size=0.75, 
    # test_size=0.25, 
    # 훈련데이터 사이즈만 잡으면 테스트 사이즈는 자동으로 설정, 통상적으로 트레인만 적는다
    # # 트레인, 테스트 사이즈 둘다 적어도 됨, 둘다 삭제하면 디폴트 값으로 지정되어 나눠준다 (디폴트 : 0.75/0.25)
    # 데이터 사이즈에 따라 7:3, 8:2, 9:1 로 조절 
    # shuffle=True, # 디폴트 섞는다.
    random_state=999, #랜덤맛집(tip!!!)
    # 훈련할때마다 값이 바뀌면 완전 다른 데이터이기 때문에 고정을 시켜야 함, 랜덤 난수표 사용, 몇번을 돌려도 동일한 데이터로 설정된다
    # ㄴ 랜덤한 놈을 동일하게 뽑아준다
    # 렌덤값만 바꿔도 성능이 잘 나올 수  있다.
)

# x_train :  [ 3  7  8  1 10  4  2]
# x_test :  [9 6 5]
# y_train :  [ 3  7  8  1 10  4  2]
# y_test :  [9 6 5]

print("x_train : ", x_train)
print("x_test : ", x_test)
print("y_train : ", y_train)
print("y_test : ", y_test)

# 정확한 판단을 위해 데이터를 분리하는 것이다.

# 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=6)

# 평가, 예측
loss = model.evaluate(x_test, y_test)
print(loss)

results = model.predict(np.array([11]))
print(results)

# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 8.7904e-13 
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step - loss: 0.0000e+00
# 0.0
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
# [[10.999999]]


# Epoch 299/300
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - loss: 0.0019
# Epoch 300/300
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step - loss: 0.0018
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 69ms/step - loss: 4.4916e-04
# 0.0004491620056796819
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step
# [[10.971082]]


# Epoch 300/300
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 4.8692e-04
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - loss: 2.6289e-04
# 0.0002628876536618918
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step
# [[10.985585]]


# shift + del : 라인삭제 단축키




