# 이미지 수치화
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

path = 'C:/study/_data/image/' # 절대경로
img = load_img(path + 'yj_1.jpeg',
               target_size=(200,200),
                             # load_img는 한장 가져올때 편함
               ) 

print(img)
print(type(img)) #<class 'PIL.Image.Image'>
# <PIL.Image.Image image mode=RGB size=150x150 at 0x2B63A529660>

# plt.imshow(img)
# plt.show()

arr = img_to_array(img) # 이미지 수치화 하는 tool
# print(arr)
# print(arr.shape) #(150, 150, 3) 3차원
# print(type(arr)) #<class 'numpy.ndarray'>

# 4차원으로 만들어 주기 위해 reshape
arr = np.expand_dims(arr, axis=0) # 차원증가
# print(arr)
# print(arr.shape) #(1, 150, 150, 3)

# 저장
# np_path = './_data/kaggle_cat_dog_npy/'
# np.save(np_path + 'keras48_my.npy', arr=arr)  # 저장 파일명 


#### 요기부터 증폭

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

### 랜덤으로 적용된다...

it = datagen.flow(arr, 
             batch_size=1,
             )

print(it)
# <keras.preprocessing.image.NumpyArrayIterator object at 0x000001AB0EA07F70>

# print(it.next()) # 파이썬 3.10까지 가능
print(next(it))  # 파이썬 3.11부터 바뀜 
print(next(it).shape)  # (1, 100, 100, 3)

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))

for i in range(5) :
    # batch = it.next() # 파이썬 3.10까지 가능, 
    batch = next(it)
    # print(batch.shape)
#     (1, 100, 100, 3)
#     (1, 100, 100, 3)
#     (1, 100, 100, 3)
#     (1, 100, 100, 3)
#     (1, 100, 100, 3)
#     (1, 100, 100, 3)
    batch = batch.reshape(200,200,3)

    ax[i].imshow(batch)
    # ax[i].axis('off')

plt.show()





