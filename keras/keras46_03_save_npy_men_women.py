# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset

# 데이터 Save


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

# 데이터 수치화
datagen = ImageDataGenerator(
    rescale=1./255,
)

# 경로 
path = './_data/image/men_women/'

xy = datagen.flow_from_directory(
    path,
    target_size=(100, 100),
    batch_size=30000,          # 전체 데이터 수보다 크게
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)
# Found 27167 images belonging to 2 classes.

# x, y 꺼내기
x = xy[0][0]
y = xy[0][1]

print(x.shape, y.shape) #(27167, 150, 150, 3) (27167,)

y= y.reshape(-1,1)
print(x.shape, y.shape) #(27167, 100, 100, 3) (27167, 1)

# from sklearn.preprocessing import OneHotEncoder

# ohe = OneHotEncoder(sparse_output=False) 
# y = ohe.fit_transform(y)
# print(y, y.shape) # ... [0. 1.]] (27167, 2)

# train / test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=908,
    shuffle=True,
    stratify=y,
)

print(x_train.shape, y_train.shape) # (21733, 100, 100, 3) (21733, 1)
print(x_test.shape, y_test.shape) # (5434, 100, 100, 3) (5434, 1)

# exit()

np_path = './_save/keras44/' 

# 저장 
np.save(np_path + 'keras44_03_men_women_x_train.npy', arr=x_train) 
np.save(np_path + 'keras44_03_men_women_y_train.npy', arr=y_train)
np.save(np_path + 'keras44_03_men_women_x_test.npy', arr=x_test)
np.save(np_path + 'keras44_03_men_women_y_test.npy', arr=y_test)

# 불러오기
# x_train = np.load(np_path + 'keras46_01_horse_x_train.npy')
# y_train = np.load(np_path + 'keras46_01_horse_y_train.npy')
# x_test = np.load(np_path + 'keras46_01_horse_x_test.npy')
# y_test = np.load(np_path + 'keras46_01_horse_y_test.npy')

exit()


# 2. 모델구성

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(150,150,3), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu'))

model.add(Flatten())
# model.add(GlobalAveragePooling2D())

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid')) #y.shape =

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam', 
              metrics=['acc'])


start_time = time.time()
model.fit(x_train, y_train,
          epochs= 30,
          batch_size=64,
          verbose=1,
          validation_split=0.2,     
          )
end_time = time.time()


# 4. 평가, 예측
results = model.evaluate(x_test, y_test,)
print('loss : ', round(results[0],3))
print('acc : ', round(results[1],3))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred,3) # 반올림 처리

acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", round(acc_score,3) ) 
print('걸린시간 : ', round(end_time-start_time,3), "초")


