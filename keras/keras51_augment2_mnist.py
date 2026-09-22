# 51-1 카피
# 36-3 카피

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

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test)= mnist.load_data()

##### 요기부터 증폭이다 ####

### 데이터 수치화 및 증폭 조건 설정
datagen = ImageDataGenerator(
    # rescale=1./255,            

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

# randidx = np.random.randint(x_train.shape[0], size=augment_size) # 중복된 데이터 있을 수 있다.
randidx = np.random.choice(x_train.shape[0], size=augment_size, replace = False) # 중복 없이 랜덤 추출

print(randidx)                            # [23683 15708 15984 ... 42898 40323 47003]
print(randidx.shape)                      # (40000,)
print(len(randidx))                       # 40000

print(np.min(randidx),np.max(randidx),)   # 0 59999

x_augmented = x_train[randidx].copy()     
y_augmented = y_train[randidx].copy()     
print(x_augmented.shape, y_augmented.shape)  # (40000, 28, 28) (40000,)

x_augmented = x_augmented.reshape(
                x_augmented.shape[0],
                x_augmented.shape[1],
                x_augmented.shape[2],1)

print(x_augmented.shape)                     # (40000, 28, 28, 1)

# x_augmented = datagen.flow(
#                 x_augmented, y_augmented,
#                 batch_size = augment_size,
#                 shuffle=False,
# ).next()[0]

x_augmented = next(
    datagen.flow(
        x_augmented, y_augmented,
        batch_size=augment_size,
        shuffle=False,
    )
)[0]

print(x_augmented.shape)                     # (40000, 28, 28, 1)
print(x_train.shape)                         # (60000, 28, 28)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
print(x_train.shape,x_test.shape ) # (60000, 28, 28, 1) (10000, 28, 28, 1)
x_train=np.concatenate((x_train, x_augmented))/255.
y_train=np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape ) # (100000, 28, 28, 1) (100000,)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)
# array([ 9770, 11272, 10050, 10364,  9733,  8968,  9814, 10550,  9636, 9843], dtype=int64))

##### 원핫인코더(분류)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False) 

y_train = y_train.reshape(-1,1) 
y_test = y_test.reshape(-1,1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape)  # (100000, 10) (10000, 10)

#2. 모델구성
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1))) 
model.add(MaxPooling2D())
model.add(Conv2D(filters=128, kernel_size=(5,5), activation='relu' )) 
model.add(Conv2D(32, (3,3), activation='relu' )) 
model.add(Conv2D(32, (3,3), activation='relu' )) 
model.add(Dropout(0.3))
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu' )) 
model.add(Conv2D(filters=16, kernel_size=(2,2), activation='relu' ))  

model.add(Flatten()) 
# model.add(GlobalAveragePooling2D())

model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))

model.add(Dense(10, activation='softmax'))  

model.summary()

# exit()

#3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer='adam',
              metrics = ['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    verbose=1,
    patience=20,
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
          epochs=80, 
          batch_size=128, 
          verbose=1,
          validation_split=0.2,
        #   callbacks = [es, mcp],
          )
end_time=time.time()

# 4. 평가, 예측
print( "=============model.evaluate=================")
loss = model.evaluate(x_test, y_test, verbose=1)

print('loss : ', round(loss[0],2))
print('acc : ', round(loss[1],2))

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print('acc_score : ', acc_score)
print ('걸린시간 : ', round(end_time-start_time,2),'초' )

##### CPU 결과
# =============model.evaluate=================
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - acc: 0.9912 - loss: 0.0371     
# loss :  0.03710708022117615
# acc :  0.9911999702453613
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step  
# acc_score :  0.9912
# 걸린시간 :  595 초

##### GPU 결과
# loss :  0.05
# acc :  0.99
# 313/313 [==============================] - 0s 1ms/step
# acc_score :  0.9899
# 걸린시간 :  136.21 초

##### 성능향상 작업 : 0.995 맞추기
## 1) EarlyStopping 추가 
# loss :  0.05
# acc :  0.99
# acc_score :  0.9908
# 걸린시간 :  114.42 초

## 2) 커널사이즈 변경 + batch_size=32 설정 (Total params: 232,906 --> 177,098  ) ******
# loss :  0.04
# acc :  0.99
# acc_score :  0.992
# 걸린시간 :  244.61 초

## 3) batch_size=256, CNN Dropout을 model.add(Dropout(0.5))로 변경, epoch = 60 증가
# loss :  0.04
# acc :  0.99
# acc_score :  0.9902
# 걸린시간 :  144.69 초

### 4) 4번째 레이어층 필터 32로 향상, Dropout 0.3 변경
# loss :  0.03
# acc :  0.99
# acc_score :  0.9927
# 걸린시간 :  122.84 초

### 4) 2번째 레이어층 필터 128로 향상, Dropout 0.5 변경
# loss :  0.03
# acc :  0.99
# acc_score :  0.9927
# 걸린시간 :  122.84 초


### 데이터 증폭 후 성능(6만장 -> 10만장 )
# loss :  6.55
# acc :  0.99
# acc_score :  0.9883
# 걸린시간 :  443.03 초

