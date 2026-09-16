import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd

(x_train, y_train), (x_test, y_test)= mnist.load_data()
# print(x_train)
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

# print(x_train[0])
# print(x_train[0][0]) #[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

print(np.unique(y_train, return_counts=True)) # y클래스 확인 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#   dtype=int64))

print(pd.value_counts(y_test))

import matplotlib.pyplot as plt
plt.imshow(x_train[9999],'gray')
plt.show()


# 문제점
# 1. 값이 너무 작으면 소멸 될 수 있다. (또 하나는 지역적인 특징만 본다는 점)
# 2. 연산량이 너무 많다.
#    ㄴ 이미지가 크고 필터 수가 많아지면 계산량과 파라미터가 빠르게 늘어나.




