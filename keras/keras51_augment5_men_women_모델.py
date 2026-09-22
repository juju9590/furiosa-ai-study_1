# 실습
# 여자 데이터 증폭해서 성능 올리기

# 이미지 수치화
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import time
import datetime
from sklearn.metrics import accuracy_score

# 1. 데이터
np_path = './_save/keras44/'

# # 저장 
# # np.save(np_path + 'keras44_03_men_women_x_train.npy', arr=x_train) 
# # np.save(np_path + 'keras44_03_men_women_y_train.npy', arr=y_train)
# # np.save(np_path + 'keras44_03_men_women_x_test.npy', arr=x_test)
# # np.save(np_path + 'keras44_03_men_women_y_test.npy', arr=y_test)



# 불러오기
x_train = np.load(np_path + 'keras44_03_men_women_x_train.npy')
y_train = np.load(np_path + 'keras44_03_men_women_y_train.npy')
x_test = np.load(np_path + 'keras44_03_men_women_x_test.npy')
y_test = np.load(np_path + 'keras44_03_men_women_y_test.npy')

print(x_train.shape, y_train.shape  ) #(21733, 100, 100, 3) (21733, 1)
print(x_test.shape, y_test.shape  ) #(5434, 100, 100, 3) (5434, 1)


y_train_1d = y_train.reshape(-1)
y_test_1d = y_test.reshape(-1)

print(y_train_1d.shape,y_test_1d.shape ) #(7591, 100, 100, 3) (7591, 1)


### 남자, 여자 분리 (0보다 큰 y값이 있는 인덱스 추출)
x_train_woman = x_train[np.where(y_train_1d >0)]
y_train_woman = y_train[np.where(y_train_1d >0)]
x_test_woman = x_test[np.where(y_test_1d >0)]
y_test_woman = y_test[np.where(y_test_1d >0)]

print(x_train_woman.shape, y_train_woman.shape  ) #(7591, 100, 100, 3) (7591, 1)
print(x_test_woman.shape, y_test_woman.shape  ) #(1898, 100, 100, 3) (1898, 1)

######## 증폭

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

augment_size = 8000

randidx = np.random.randint(x_train_woman.shape[0], size=augment_size,) # replace=False) # 중복없이 랜덤 선택
print(randidx.shape) #(8000,)

print(np.min(randidx),np.max(randidx),) # 0 7590

x_augmented_woman = x_train_woman[randidx].copy() 
y_augmented_woman = y_train_woman[randidx].copy()

print(x_augmented_woman.shape, y_augmented_woman.shape) #(8000, 100, 100, 3) (8000, 1)

### x_augmented_woman 3차원을 4차원 형태로 리쉐이프
x_augmented_woman = x_augmented_woman.reshape(
                x_augmented_woman.shape[0],
                x_augmented_woman.shape[1],
                x_augmented_woman.shape[2],3)

print(x_augmented_woman.shape) #(8000, 100, 100, 3)

### x 데이터만 변환
x_augmented_woman = datagen.flow(
                x_augmented_woman, y_augmented_woman,
                batch_size = augment_size,
                shuffle=False,
).next()[0]

print(x_augmented_woman.shape) #(8000, 100, 100, 3)
print(x_train_woman.shape) #(7591, 100, 100, 3)

x_train_woman = x_train_woman.reshape(
        x_train_woman.shape[0],
        x_train_woman.shape[1],
        x_train_woman.shape[2],3)

x_test_woman = x_test_woman.reshape(
        x_test_woman.shape[0],
        x_test_woman.shape[1],
        x_test_woman.shape[2],3)

print(x_train_woman.shape, x_test_woman.shape ) #

### 트레인과 증폭된 이미지 붙이기 (여자만)
x_train_woman=np.concatenate((x_train_woman, x_augmented_woman))/255.
y_train_woman=np.concatenate((y_train_woman, y_augmented_woman))

print(x_train_woman.shape, y_train_woman.shape )
# (21733, 100, 100, 3) (5434, 100, 100, 3)
# (15591, 100, 100, 3) (15591, 1)
print(np.unique(y_train_woman, return_counts=True))
# (array([1.], dtype=float32), array([15591], dtype=int64))

### 남자+여자 