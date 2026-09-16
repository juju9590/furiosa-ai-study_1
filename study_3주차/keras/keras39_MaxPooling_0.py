# 38 copy


import numpy as np
import pandas as pd

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D

#2. 모델구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1), 
                 padding='same', 
                 strides=1, 
                 ))

model.add(MaxPooling2D()) # Conv2D 다음에 쓴다 

model.add(Conv2D(filters=9, kernel_size=(3,3), 
                 padding='valid',
                 strides=2,
                 ))

model.summary()