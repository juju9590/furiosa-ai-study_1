from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))


# x_temp, x_test, y_temp, y_test = train_test_split(
#     x, y,
#     test_size=0.5,
#     random_state= 42,
#     # shuffle=True,
#     # stratify=y,
# )

# x_train, x_val, y_train, y_val = train_test_split(
#     x_temp, y_temp,
#     test_size=0.2,
#     random_state= 42,
#     # shuffle=True,
#     # stratify=y_temp,
# )

# print(x_temp.shape, x_val.shape, x_test.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,test_size=0.5, random_state= 42, )

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test,test_size=0.5, random_state= 42, )

print(x_train)
print(x_test)
print(x_val)

# 2. 모델구성
model = Sequential()



