# y햇 = wx + b (예측, predict) = y_predict   w, b 는 train set으로 이미 구함
#     = w * x_test + b
# loss = (y_test - y_predict)^2 -> 평균
# 그래서 evaluate에 x_test, y_test 값을 넣음

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import numpy as np

# 1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
print(y_train.shape, y_test.shape) # (404,) (102,)

# 2. 모델 구성
model = Sequential()
model.add(Dense(7, input_dim=13))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 학습
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=1500, batch_size=64)

print("=====================================================")
# 4. 평가, 예측
loss = model.evaluate(x_test, y_test) # evaluate에는 predict 이 포함이 되어있음
print("loss : ", loss)

y_predict = model.predict(x_test) # 이렇게 하고 원값 y_test랑 예측값 y_predict를 비교함
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
r2 = r2_score(y_test, y_predict) # 원값 y_test랑 예측값 y_predict를 비교
print("r2 : ", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse :", mse)

def RMSE(y_test, y_predict): # RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
    

# loss(mse) :  24.763097763061523
# r2 :  0.722523397161407

# loss :  29.84109115600586
# r2 :  0.6415219573685786
# mse : 29.841091878484477
# RMSE :  5.462700053863884