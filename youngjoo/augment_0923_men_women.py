# 데이터 불균형 해소


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
start_data = time.time()

np_path = './_save/keras44/' 

# np.save(np_path + 'keras44_0923_men_women_x_train.npy', arr=x_train)
# np.save(np_path + 'keras44_0923_men_women_y_train.npy', arr=y_train)
# np.save(np_path + 'keras44_0923_men_women_x_test.npy', arr=x_test)
# np.save(np_path + 'keras44_0923_men_women_y_test.npy', arr=y_test)

# 불러오기
x_train = np.load(np_path +'keras44_0923_men_women_x_train.npy')
y_train = np.load(np_path +'keras44_0923_men_women_y_train.npy')
x_test = np.load(np_path +'keras44_0923_men_women_x_test.npy')
y_test = np.load(np_path +'keras44_0923_men_women_y_test.npy')

print(x_train.shape, y_train.shape) #(21733, 100, 100, 3) (21733, 1)
print(x_test.shape, y_test.shape) #(5434, 100, 100, 3) (5434, 1)


### 남자, 여자 분리 


end_data = time.time()
print("데이터 걸린시간 : ", round(end_data-start_data,3),"초") # 2.036 초