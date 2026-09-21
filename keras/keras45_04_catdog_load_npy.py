# 44-1 카피

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D

import time
from sklearn.metrics import accuracy_score

# 1. 데이터

train_datagen = ImageDataGenerator(
    rescale=1./255,            

    # horizontal_flip=True,       # 수평 뒤집기,
    # vertical_flip=True,         # 수직 뒤집기,
    # width_shift_range=0.1,      # 평형이동
    # height_shift_range=0.1,
    # rotation_range=5,           # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,             # 확대
    # shear_range=0.7,            # 좌표하나를 고정하고 다른 몇개의 좌표를 이동(한마디로 찌부) 
    # fill_mode='nearest',        # 변화나 이동으로 인해 없어진 값은 근처의 값으로 채운다
)  

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

# 파일의 경로
path_train = './_data/image/cat_dog/training_set/'  
path_test = './_data/image/cat_dog/test_set/'

xy_train = train_datagen.flow_from_directory(            
    path_train,                 # 폴더의 경로
    target_size=(150, 150),     # 이미지 크기 동일하게 맞추기
    batch_size=10000,             
    class_mode='binary',        # 이진분류
    color_mode='rgb',     # 흑백
    shuffle=True,
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(150, 150),     
    batch_size=10000,              
    class_mode='binary',        
    color_mode='rgb',     
    shuffle=False, # test에서 필요가 없다 
)
# Found 120 images belonging to 2 classes

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

# print(x_train.shape, y_train.shape) #(160, 150, 150, 1) (160,)
# print(x_test.shape, y_test.shape)   #(120, 150, 150, 1) (120,)

### 데이터 변환 시간 오래 걸리니, 

np_path = './_data/kaggle_cat_dog_npy/'

# np.save(np_path + 'keras45_02_cat_dog_x_train.npy', arr=xy_train[0][0]) # x_train 
# np.save(np_path + 'keras45_02_cat_dog_y_train.npy', arr=xy_train[0][1]) # y_train 
# np.save(np_path + 'keras45_02_cat_dog_x_test.npy', arr=xy_test[0][0]) # x_test 
# np.save(np_path + 'keras45_02_cat_dog_y_test.npy', arr=xy_test[0][1]) # y_test


x_train = np.load(np_path + 'keras45_02_cat_dog_x_train.npy')
y_train = np.load(np_path + 'keras45_02_cat_dog_y_train.npy')
x_test = np.load(np_path + 'keras45_02_cat_dog_x_test.npy')
y_test = np.load(np_path + 'keras45_02_cat_dog_y_test.npy')

print(x_train.shape, y_train.shape) #(8005, 150, 150, 3) (8005,)
print(x_test.shape, y_test.shape)   #(2023, 150, 150, 3) (2023,)


exit()