# 변환된 데이터 불러오기
# 모델구성 컴파일/훈련 => 모델 저장하기
# 


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import time
import datetime
from sklearn.metrics import accuracy_score

#1. 데이터 (save_0923_men_women.py)

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

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(100,100,3), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu'))

# model.add(Flatten())
model.add(GlobalAveragePooling2D())

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid')) #y.shape =

model.summary()

#3. 컴파일, 훈련
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

### 전체 모델 저장 
path ='./_save/keras49/'
filename = 'keras49_save_model_0923.keras'
model.save(path + filename)

#4. 평가, 예측
results = model.evaluate(x_test, y_test,)
print('loss : ', round(results[0],3))
print('acc : ', round(results[1],3))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred) # binary 반올림 처리

acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", round(acc_score,3) ) 
print('걸린시간 : ', round(end_time-start_time,3), "초")


# 결과
# loss :  0.267
# acc :  0.888

# loss :  0.254
# acc :  0.896
