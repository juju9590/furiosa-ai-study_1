# 이미지 수치화
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt

path = 'C:/study/_data/image/' # 절대경로
img = load_img(path + 'yj_1.jpeg',
               target_size=(100,100),
                             # load_img는 한장 가져올때 편함
               ) 

print(img)
print(type(img)) #<class 'PIL.Image.Image'>
# <PIL.Image.Image image mode=RGB size=150x150 at 0x2B63A529660>

# plt.imshow(img)
# plt.show()

arr = img_to_array(img) # 이미지 수치화 하는 tool
print(arr)
print(arr.shape) #(150, 150, 3) 3차원
print(type(arr)) #<class 'numpy.ndarray'>

# 4차원으로 만들어 주기 위해 reshape
arr = np.expand_dims(arr, axis=0) # 차원증가
print(arr)
print(arr.shape) #(1, 150, 150, 3)

np_path = './_data/kaggle_cat_dog_npy/'

np.save(np_path + 'keras48_my.npy', arr=arr)  # 저장 파일명 

##################################

# img111 = load_img(path + 'tori_1.jpeg',
#                target_size=(100,100),
#                              # load_img는 한장 가져올때 편함
#                ) 

# print(img111)
# print(type(img111)) #<class 'PIL.Image.Image'>
# # <PIL.Image.Image image mode=RGB size=150x150 at 0x2B63A529660>

# # plt.imshow(img)
# # plt.show()

# arr111 = img_to_array(img111) # 이미지 수치화 하는 tool
# print(arr111)
# print(arr111.shape) #(150, 150, 3) 3차원
# print(type(arr111)) #<class 'numpy.ndarray'>

# # 4차원으로 만들어 주기 위해 reshape
# arr111 = np.expand_dims(arr111, axis=0) # 차원증가
# print(arr111)
# print(arr111.shape) #(1, 150, 150, 3)

# np_path = './_data/kaggle_cat_dog_npy/'

# np.save(np_path + 'keras48_tori.npy', arr=arr111)  # 저장 파일명 