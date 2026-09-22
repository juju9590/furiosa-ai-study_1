# 50-1 copy


# 이미지 수치화
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train),(x_test, y_test)=fashion_mnist.load_data()

##### 요기부터 증폭이다 ####
### 이미지 1개를 100장으로 증폭 시키는 방법 


datagen = ImageDataGenerator(
    rescale=1./255,            

    horizontal_flip=True,       # 수평 뒤집기,
    # vertical_flip=True,         # 수직 뒤집기, (상하반전)
    width_shift_range=0.2,      # 평형이동
    # height_shift_range=0.1,
    rotation_range=5,           # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range=0.1,             # 확대
    # shear_range=0.7,            # 좌표하나를 고정하고 다른 몇개의 좌표를 이동(한마디로 찌부) 
    fill_mode='nearest',        # 변화나 이동으로 인해 없어진 값은 근처의 값으로 채운다
)  

augment_size = 100 # augment : 증가시키다

print(x_train.shape) #(60000, 28, 28)
print(x_train[0].shape) #(28, 28) 6만장 중 1장 

# aaa = np.tile(x_train[0], augment_size)  #tile (타일) : 붙이다
# np.tile은 1차원 데이터 형태로 표현
# print(aaa.shape)  # (28, 2800)
# ㄴ 현재는 똑같은 1장을 100번 붙이기 (증폭)
# ㄴ 나중에 증폭시에는 변형시켜서 100개 증폭 해야 함 
# ㄴ 이건 그냥 이어붙인거기 때문에 한장씩 잘라줘야 한다 

aaa = np.tile(x_train[0], augment_size).reshape(-1,28,28,1)
print(aaa.shape) #(100, 28, 28, 1)
# 요것도 단순 복붙이고 한장씩 나눠 놓은것이다

xy_data = datagen.flow(
        np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),
        # np.tile은 1차원의 데이터로 x_train[0].reshape(28*28) 쫙 펼치고 , 다시 .reshape(-1,28,28,1), 해줌
        np.zeros(augment_size),
        batch_size=augment_size, #통배치
        shuffle=False,
).next()
# datagen = 위에서 ImageDataGenerator로 변환한 값을 반환한 값
# 위 내용은 x와 y가 모여있는 이터레이터 데이터 
# 블록지정 + tap / shift+tap

print(xy_data)
print(type(xy_data)) #<class 'tuple'>
# print(xy_data.shape) #AttributeError: 'tuple' object has no attribute 'shape'
print(len(xy_data))  # 2 => 왜냐하면 x, y 2개니깐
print(xy_data[0].shape) #(100, 28, 28, 1)
print(xy_data[1].shape) #(100,)

# 1장의 사진을 100장을 증폭 시켰고

plt.figure(figsize=(10,10))
for i in range(100) :
    plt.subplot(10,10,i+1)
    plt.imshow(xy_data[0][i], cmap='gray')
plt.show()

# 그림으로는  100장 중  49장 살펴보기


## 6만장의 데이터에서  4만장 증폭시켜서 총 10만장 만들기 
## 6만장 중 4만장 뽑아서 1회 데이터 변환 시킨 후 합치면 (붙이다) 10만장 확보







