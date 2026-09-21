# 9-1 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train =np.array([1,2,3,4,5,6])
y_train =np.array([1,2,3,4,5,6])

x_val =np.array([7,8])
y_val =np.array([7,8])

x_test =np.array([9,10])
y_test =np.array([9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=4, 
        verbose=1,
        validation_data=(x_val, y_val)
          )
# verbose=0 : 침묵
# verbose=1 : 디폴트
# verbose=2 : 프로그래스바 삭제
# verbose= 나머지 : 에포만 나옴
# 인공지능은 전기세, 속도, 성능이 중요한데 
# 1 epochs 돌릴때 강제 딜레이가 되기 때문에 0,1,2를 상황에 맞게 선택한다.
# 시간이 오래 걸리는 모델은 verbose=1 로 진행과정을 보는게 좋고, 
# 빠르게 돌아가는건 0, 2를 선택하는게 좋다.

print("===================================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)