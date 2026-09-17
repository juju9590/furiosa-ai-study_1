# 33-9
# DNN -> CNN 으로 변경 

from sklearn.datasets import fetch_covtype # 시간체크, 배치사이즈 크게 하기

# acc = 0.93 이상

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = fetch_covtype()
# print(datasets.DESCR)

x = datasets.data
y = datasets.target

print(x.shape, y.shape)

print(np.unique(y, return_counts=True)) 
#(array([1, 2, 3, 4, 5, 6, 7], dtype=int32), 
# array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

### 원핫인코딩
y = pd.get_dummies(y, dtype=int)
print(y.shape) #(581012, 7)

### train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=y,
                                                    )

### scaler (x_train, x_test 만)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test) 

print(np.min(x_train), np.max(x_train)) # -9.416666666666666 9.17741935483871
print(np.min(x_test), np.max(x_test))   # -9.416666666666666 9.209677419354838

############ X 데이터를 CNN에 넣기 위해 4차원으로 변환
x_train = x_train.reshape(-1,9,6,1)
x_test = x_test.reshape(-1,9,6,1)

print(x_train.shape, x_test.shape) #(464809, 54, 1, 1) (116203, 54, 1, 1)
print(y_train.shape, y_test.shape) #(464809, 7) (116203, 7)

#2. 모델구성
model = Sequential()

model.add(Conv2D(64, (2,2), input_shape=(9,6,1))) 
model.add(Conv2D(32, (2,2), padding='same', strides=2, activation='relu' ))
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2), padding='same', activation='relu' ))

# model.add(Flatten())
model.add(GlobalAveragePooling2D())

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(7, activation='softmax'))  

model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'],)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    restore_best_weights=True,
    patience=200,
    verbose=1,
)

import datetime
date = datetime.datetime.now() 
print(date) 
print(type(date)) 
date = date.strftime("%m%d_%H%M") 
print(date) #
print(type(date)) 

path ='./_save/keras31/'

filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k31_09_", date, "-" ,filename])

 
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath,
    verbose=1,
)

start_time=time.time()
model.fit(x_train, y_train,
          epochs=80,
          batch_size=128,
          verbose=1,
          validation_split=0.2,
          callbacks=[es,mcp],
          )
end_time=time.time()

#4. 예측, 평가
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc :", result[1])

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred )
print("acc_score :", acc_score)
print("걸린시간 :", round(end_time-start_time,3),"초")



#### 결과 save
# loss : 0.5519099235534668
# acc : 0.7656514644622803
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 413us/step 
# acc_score : 0.7656514892042374

# epochs=50
########### GPU
# 걸린시간 : 258.822 초

########### CPU 
# 걸린시간 : 143.341 초 

#### dnn >>> cnn 1차 기본
# loss : 0.2742099463939667
# acc : 0.8885054588317871
# acc_score : 0.88850546027211
# 걸린시간 : 805.006 초

#### dnn >>> cnn 2차 
# loss : 0.40213915705680847
# acc : 0.8285242319107056
# acc_score : 0.8285242205450806
# 걸린시간 : 1111.718 초