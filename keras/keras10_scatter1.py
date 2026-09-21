import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])

# [검색] train과 test를 섞어서 7:3 나눈다
# 힌트 : 사이킷런

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    # train_size=0.75, 
    # test_size=0.25,  
    # shuffle=True, # 디폴트 섞는다.
    random_state=999, #랜덤맛집(tip!!!)

)


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
model.fit(x_train, y_train, epochs=100, batch_size=6)

print("============================================")

# 평가, 예측
loss = model.evaluate(x_test, y_test)
print(loss)
# 학습에서 적용된 W로 딱 한번만 할 수 있다.

results = model.predict(x)
print(results)

# 그래프 그리기
import matplotlib.pyplot as plt
plt.scatter(x, y) #데이터 점찍기 (산점도)
plt.plot(x, results, color='red') # 선 그리기
plt.show() # 그래프 보기




