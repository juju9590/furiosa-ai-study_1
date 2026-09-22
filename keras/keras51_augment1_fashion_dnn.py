# 50-2 copy
# 6만장에서 4만장 증폭해서 10만장 만들기

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import time
import datetime
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train),(x_test, y_test)=fashion_mnist.load_data()

##### 요기부터 증폭이다 ####

### 데이터 수치화 및 증폭 조건 설정
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

### 증폭할 사이즈 변수 선언 
augment_size = 40000 

# print(x_train.shape[0]) #60000  #x_train.shape(60000,28,28,1)에서 첫번째 수를 가져오고 싶음 [0] 붙여줌

# randidx = np.random.randint(60000, size=augment_size) # 데이터의 갯수를 알때
randidx = np.random.randint(x_train.shape[0], size=augment_size) # 데이터의 갯수를 몰를때.. 또는 통상
# 총 x_train데이터에서 랜덤으로 4만장만 선정한걸 randidx 변수로 반환 

print(randidx) # 백터 [30827 53243 35516 ... 10309 47106 54572]
print(randidx.shape) # 백터니깐 shape 가능 (40000,)
print(len(randidx)) # 리스트랑 튜플은 len으로 확인, 백터도 확인가능 40000

print(np.min(randidx),np.max(randidx),) # 1 59998 (4만개를 선정할때 최대값과 최소값의 범위를 임의로 선정, 0과 59999는 선택이 안 되었을 뿐)

x_augmented = x_train[randidx].copy() # 기존의 변수와 영향이 없도록 하기 위헤 .copy() 처리
y_augmented = y_train[randidx].copy()
# 랜덤으로 선정된 x_train중 4만장을 x_augmented로 지정 , y도 동일 
print(x_augmented.shape, y_augmented.shape) #(40000, 28, 28) (40000,)

# x_augmented = x_augmented.reshape(40000,28,28,1)

x_augmented = x_augmented.reshape(
                x_augmented.shape[0],
                x_augmented.shape[1],
                x_augmented.shape[2],1)

print(x_augmented.shape) #(40000, 28, 28, 1)

x_augmented = datagen.flow(
                x_augmented, y_augmented,
                batch_size = augment_size,
                shuffle=False,
).next()[0]
# x 만 데이터 변환하고 y값은 그대로 가져오면 된다.. 그래서 .next()[0]을 설정하고 x_augmented로 설정

print(x_augmented.shape) #(40000, 28, 28, 1)
print(x_train.shape) #(60000, 28, 28)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
print(x_train.shape,x_test.shape ) #(60000, 28, 28, 1) (10000, 28, 28, 1)

x_train=np.concatenate((x_train, x_augmented))/255.
y_train=np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape ) #(100000, 28, 28, 1) (100000,)

print(np.unique(y_train, return_counts=True)) 
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([ 9956,  9965, 10019, 10044,  9934, 10001, 10006,  9996,  9993, 10086], dtype=int64))

# [실습] 조건 : acc 0.94 이상


### 스케일링 (x_train 만)
x_train = x_train/255
x_test = x_test/255

### x-reshape (4차원을 2차원으로 변환)
x_train = x_train.reshape(-1,28*28*1)
x_test = x_test.reshape(-1,28*28*1)

print(x_train.shape,x_test.shape) #(100000, 784) (10000, 784)


# exit()

##### 원핫인코더(분류)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False) 

y_train = y_train.reshape(-1,1) 
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape) 

#2. 모델구성
model = Sequential()
model.add(Dense(256, input_shape=(28*28,), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))

model.add(Dense(10, activation='softmax'))  

model.summary()

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer='adam',
              metrics = ['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    verbose=1,
    patience=200,
)

import datetime
date = datetime.datetime.now()
date = date.strftime('%d%m-%H%M')

path ='./_save/keras36/'

filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k36_", date, "-", filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    save_best_only=True,
    verbose=1,
    filepath=filepath,
    mode='min',
)

start_time=time.time()
model.fit(x_train,y_train,
          epochs=70, 
          batch_size=64, 
          verbose=1,
          validation_split=0.2,
        #   callbacks = [es, mcp],
          )
end_time=time.time()

# 4. 평가, 예측
print( "=============model.evaluate=================")
loss = model.evaluate(x_test, y_test, verbose=1)

print('loss : ', round(loss[0],3))
print('acc : ', round(loss[1],3))

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print('acc_score : ', acc_score)
print ('걸린시간 : ', round(end_time-start_time,2),'초' )

### CNN

### 결과 5 MaxPooling2D 추가 (플래튼)
# loss :  0.251
# acc :  0.912
# acc_score :  0.9119
# 걸린시간 :  262.39 초

### 결과 5 GlobalAveragePooling2D
# loss :  0.24
# acc :  0.917
# acc_score :  0.9172
# 걸린시간 :  172.24 초


### DNN 1차
# loss :  0.349
# acc :  0.883
# acc_score :  0.883
# 걸린시간 :  39.66 초

### DNN 2차
# loss :  0.381
# acc :  0.873
# acc_score :  0.8725
# 걸린시간 :  43.54 초

### DNN 3차
# loss :  0.489
# acc :  0.89
# acc_score :  0.8897
# 걸린시간 :  127.78 초

### 1차 (1만장 증폭 후 성능)
# 313/313 [==============================] - 0s 1ms/step - loss: 312.3601 - acc: 0.6079
# loss :  312.36
# acc :  0.608
# 313/313 [==============================] - 0s 993us/step
# acc_score :  0.6079
# 걸린시간 :  249.6 초
