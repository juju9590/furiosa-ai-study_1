#36-2 copy

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd

#1. 데이터
(x_train, y_train), (x_test, y_test)= mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

print(np.max(x_train), np.min(x_train)) # 255 0
print(np.max(x_test), np.min(x_test)) # 255 0
# 통상 이미지의 X데이터는 0~255로 구성

##### 스케일링 1
# x_train = x_train/255. # .만 붙이면 float 형태로 출력하게 됨
# x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) # 1.0 0.0 (0~1 사이로 나옴)
# print(np.max(x_test), np.min(x_test)) # 1.0 0.0
# # MinMaxScaler, MaxAbsScaler 와 동일, 이미지기 때문에 255로 나누면 동일한 값이 나온다

##### 스케일링 2
x_train = (x_train-127.5)/127.5
x_test = (x_test-127.5)/127.5
print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
print(np.max(x_test), np.min(x_test)) # 1.0 -1.0
# 굳이 MinMaxScaler, MaxAbsScaler 이런 스케일링을 하지 않아도 위와 같이 하면 동일한 효과를 볼 수 있다.













