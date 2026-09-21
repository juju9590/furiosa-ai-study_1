'''
개 고양이 가중치를 가져와서 모델 완성 (#2)
데이터는 개 고양이 npy 데이터 사용 (#1)
내 사진도 npy 불러와서 predict 해보기 (#4)
'''

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import time
import datetime
from sklearn.metrics import accuracy_score


# 1. 데이터

np_path = './_data/kaggle_cat_dog_npy/'

x_train = np.load(np_path + 'keras45_02_cat_dog_x_train.npy')
y_train = np.load(np_path + 'keras45_02_cat_dog_y_train.npy')
x_test = np.load(np_path + 'keras45_02_cat_dog_x_test.npy')
y_test = np.load(np_path + 'keras45_02_cat_dog_y_test.npy')

my_photo = np.load(np_path + 'keras48_my.npy')
tori_photo = np.load(np_path + 'keras48_tori.npy')

## 스케일링
my_photo = my_photo/255.
tori_photo = tori_photo/ 255

print(np.max(my_photo), np.min(my_photo))  # 1.0 0.011764706
print(np.max(tori_photo), np.min(tori_photo)) #1.0 0.0

# exit()

# 2. 모델구성 + #3. 컴파일, 훈련

path ='./_save/keras44_model/'
model = load_model(path + 'keras44_cat_dog_save_model.keras')


# 4. 성능, 예측
results = model.evaluate(x_test, y_test,)
print('loss : ', round(results[0],4))
print('acc : ', round(results[1],4))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) # 반올림 처리


acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", acc_score ) 
# print('걸린시간 : ', round(end_time-start_time,3), "초")

print("=========== 내 사진 예측===============")

y_pred_me = model.predict(my_photo)
y_pred_me = np.round(y_pred_me,4) # 반올림 처리

print(y_pred_me, y_pred_me.shape ) #[[0.8307]] (1, 1)


print("=========== 토리 예측===============")

y_pred_tori = model.predict(tori_photo)
y_pred_tori = np.round(y_pred_tori,4) # 반올림 처리

print(y_pred_tori,y_pred_tori.shape ) #[[0.6524]] (1, 1)


### 3대장 tree
###  XGboost, LGboost, Catboost







