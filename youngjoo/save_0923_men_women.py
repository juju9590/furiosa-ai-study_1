
# 폴더에서 데이터 불러오기
# 데이터 분리
# 스케일링 + 사이즈 맞추기
# 변환된 데이터 저장(npy)


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
start_data = time.time()

datagen=ImageDataGenerator(
    rescale=1./255
)

path = './_data/image/men_women/'

xy = datagen.flow_from_directory(
    path,
    target_size=(100,100),
    batch_size=30000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)

x = xy[0][0]
y = xy[0][1]
print(x.shape, y.shape) #(27167, 100, 100, 3) (27167,)

###
# xy[0] : (x데이터, y데이터)
# xy[0][0] : x데이터
# xy[0][1] : y데이터

y = y.reshape(-1,1)
print(y.shape)  #(27167, 1) ==> 1차원에서 2차원 데이터로 변환 

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=777,
    shuffle=True,
    stratify=y, #y의 클래스 분포를 기준으로 train/test를 비슷한 비율로 나눠라
)

print(x_train.shape, y_train.shape) #(19016, 100, 100, 3) (19016, 1)
print(x_test.shape, y_test.shape) #(8151, 100, 100, 3) (8151, 1)

end_data = time.time()
print("데이터 걸린시간 : ", round(end_data-start_data,3),"초") # 27.261 초


np_path = './_save/keras44/' 

np.save(np_path + 'keras44_0923_men_women_x_train.npy', arr=x_train)
np.save(np_path + 'keras44_0923_men_women_y_train.npy', arr=y_train)
np.save(np_path + 'keras44_0923_men_women_x_test.npy', arr=x_test)
np.save(np_path + 'keras44_0923_men_women_y_test.npy', arr=y_test)

