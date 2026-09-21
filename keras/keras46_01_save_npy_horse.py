import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential
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
path = './_data/image/horse-human/'

xy = datagen.flow_from_directory(
    path,
    target_size=(150, 150),
    batch_size=1500,          # 전체 데이터 수보다 크게
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
)
# Found 1027 images belonging to 2 classes.

# x, y 꺼내기
x = xy[0][0]
y = xy[0][1]

print(x.shape, y.shape) #(1027, 150, 150, 3) (1027, 2)


# stratify용 정수 라벨
y_stratify = np.argmax(y, axis=1)

# train / test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=908,
    shuffle=True,
    stratify=y_stratify,
)

print(x_train.shape, y_train.shape) #(718, 150, 150, 3) (718, 2)
print(x_test.shape, y_test.shape) #(309, 150, 150, 3) (309, 2)

np_path = './_save/keras46_01/' # C:\study\_save\keras46_01

# 저장 
# np.save(np_path + 'keras46_01_horse_x_train.npy', arr=x_train) 
# np.save(np_path + 'keras46_01_horse_y_train.npy', arr=y_train)
# np.save(np_path + 'keras46_01_horse_x_test.npy', arr=x_test)
# np.save(np_path + 'keras46_01_horse_y_test.npy', arr=y_test)

# 불러오기
x_train = np.load(np_path + 'keras46_01_horse_x_train.npy')
y_train = np.load(np_path + 'keras46_01_horse_y_train.npy')
x_test = np.load(np_path + 'keras46_01_horse_x_test.npy')
y_test = np.load(np_path + 'keras46_01_horse_y_test.npy')

# exit()


# 2. 모델구성

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(150,150,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Dropout(0.2))


model.add(Flatten())
# model.add(GlobalAveragePooling2D())

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(2, activation='softmax')) #y.shape =(1027, 2)

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam', 
              metrics=['acc'])


start_time = time.time()
model.fit(x_train, y_train,
          epochs= 50,
          batch_size=32,
          verbose=1,
          validation_split=0.2,     
          )
end_time = time.time()


# exit()

# 4. 평가, 예측
results = model.evaluate(x_test, y_test,)
print('loss : ', round(results[0],3))
print('acc : ', round(results[1],3))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) # 반올림 처리

acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", acc_score ) 
print('걸린시간 : ', round(end_time-start_time,3), "초")