# 이제는 말 할 수 있다.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim = 1)) # x = 1 (열)
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1)) # y = 1 , 회귀모델 (왜? activation = 디폴트 니깐)

model.summary() # 파라미터의 갯수를 세기 위해서 만듦

# model.add(Dense(3, input_dim = 1)) ### 파라미터 1*3 + 3 = 6    ==> y = xw +b 
# model.add(Dense(4))                ### 파라미터 3*4 + 4 = 16   ==> y = xw +b 
# model.add(Dense(3))                ### 파라미터 4*3 + 3 = 15   ==> y = xw +b 
# model.add(Dense(1))                ### 파라미터 3*1 + 1 = 4    ==> y = xw +b 

# y 원래 식은 wx+b가 아니라 xw+b가 맞다 ==> 편의를 위해서 후자로 언급하는것..
# 행렬연산 이기때문에 X, W가 순서가 바뀌면 완전히 다른 값이 된다.

# Total params: 41 (164.00 B)
# Trainable params: 41 (164.00 B)
# Non-trainable params: 0 (0.00 B)







