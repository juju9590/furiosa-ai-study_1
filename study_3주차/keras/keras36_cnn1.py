from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D
# Conv2D : 자른 필터는 가로*세로 만 있으니 2D , 이미지를 자르는 것을 Conv2D 라고 생각하면 쉽다

model = Sequential()
model.add(Conv2D(10,(3,3),input_shape=(10,10,1))) #행무시열우선 (10000,5,5,1) => 데이터갯수를 뺀 나머지가 특성이 된다
model.add(Conv2D(5,(2,2))) # 커널사이즈 계산법 : 10-K(커널사이즈)+1 ==> 10-3+1 = 8

model.summary()

#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 8, 8, 10)          100       
#  conv2d_1 (Conv2D)           (None, 7, 7, 5)           205      
# =================================================================
# Total params: 305
# Trainable params: 305
# Non-trainable params: 0
# 오늘은 커널사이즈의 개념까지만 알자구, 파라미터계산는 내일