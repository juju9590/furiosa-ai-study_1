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
path_train = './_data/image/brain/train/'  #C:\study\_data\image\brain\train
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(            
    path_train,                 # 폴더의 경로
    target_size=(150, 150),     # 이미지 크기 동일하게 맞추기
    batch_size=160,             
    class_mode='binary',        # 이진분류
    color_mode='grayscale',     # 흑백
    shuffle=True,
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(150, 150),     
    batch_size=120,              
    class_mode='binary',        
    color_mode='grayscale',     
    shuffle=False, # test에서 필요가 없다 
)
# Found 120 images belonging to 2 classes

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape) #(160, 150, 150, 1) (160,)
print(x_test.shape, y_test.shape)   #(120, 150, 150, 1) (120,)

# 실습 : acc 1.0
# 2. 모델구성

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(150,150,1), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))

model.add(Flatten())
# model.add(GlobalAveragePooling2D())

model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# exit()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

start_time = time.time()
model.fit(x_train, y_train,
          epochs= 150,
          batch_size=32,
          verbose=1,
          validation_split=0.2,          
          )
end_time = time.time()

# 4. 평가, 예측
results = model.evaluate(x_test, y_test,)
print('loss : ', round(results[0],3))
print('acc : ', round(results[1],3))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) # 반올림 처리

acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", acc_score ) 
print('걸린시간 : ', round(end_time-start_time,3), "초")


### 결과
# loss :  0.006
# acc :  1.0
# acc_score :  1.0

### 결과
# loss :  0.075
# acc :  0.975
# acc_score :  0.975
# 걸린시간 :  38.064 초

### 
# loss :  0.011
# acc :  1.0
# acc_score :  1.0
# 걸린시간 :  55.225 초






