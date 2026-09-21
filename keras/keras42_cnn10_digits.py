# 33-10 카피

from sklearn.datasets import load_digits

# acc = 1.0

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
# print(datasets.DESCR)
print(x.shape, y.shape) #(1797, 64) (1797,)

from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1, 1) # 1차원의 데이터를 2차원의 데이터로 변환
# print(y)

ohe = OneHotEncoder(sparse_output=False) #원핫인코더 정의 + 넘파이 배열로 나타내기 위해 sparse_output=False
y = ohe.fit_transform(y)
# print(y, y.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                    test_size=0.2,
                                    random_state=333,
                                    shuffle=True,
                                    stratify=y,

                                    )

print(x_train.shape, x_test.shape) # (1437, 64) (360, 64)
print(y_train.shape, y_test.shape) # (1437, 10) (360, 10)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler() # 1번
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train) 

x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) # -2.6 16.0
print(np.min(x_test), np.max(x_test)) # -2.6 16.0

########### X 데이터를 CNN에 넣기 위해 4차원으로 변환
x_train = x_train.reshape(-1,8,8,1)
x_test = x_test.reshape(-1,8,8,1)

print(x_train.shape, x_test.shape) # (1437, 8, 8, 1) (360, 8, 8, 1)
print(y_train.shape, y_test.shape) # (1437, 10) (360, 10)

#2. 모델구성
model = Sequential()

model.add(Conv2D(64, (2,2), input_shape=(8, 8, 1))) 
model.add(Conv2D(32, (2,2) , activation='relu' )) 
model.add(MaxPooling2D())
model.add(Conv2D(16, (2,2) , padding='same', activation='relu' )) 

model.add(GlobalAveragePooling2D())
# model.add(Flatten())

# model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'],)

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=20,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs=1000,
          batch_size=4,
          validation_split=0.2,
          callbacks=[es],
          )
end_time = time.time()

print("걸린시간 :", round(end_time-start_time,2), "초")

print("================= 훈련 종료 =====================")

#4. 예측, 평가
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc :", result[1])

y_pred = np.argmax(model.predict(x_test), axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score :", acc_score)

####################### RobustScaler  ############## ====> 성능하향
# Epoch 41/1000
# 288/288 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - acc: 1.0000 - loss: 3.3231e-06 - val_acc: 0.9826 - val_loss: 0.0996
# 걸린시간 : 13.71 초
# ================= 훈련 종료 =====================
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.9583 - loss: 0.1626  
# loss : 0.1625954806804657
# acc : 0.9583333134651184
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
# acc_score : 0.9583333333333334


########## 드랍아웃 ==> 향상
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.9750 - loss: 0.1199 
# loss : 0.11985522508621216
# acc : 0.9750000238418579
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
# acc_score : 0.975

####### dnn >>> cnn (1차)
# 걸린시간 : 23.8 초
# loss : 0.07539817690849304
# acc : 0.980555534362793
# acc_score : 0.9805555555555555

####### dnn >>> cnn (2차)
# 걸린시간 : 40.9 초
# loss : 0.11017653346061707
# acc : 0.9666666388511658
# acc_score : 0.9666666666666667

####### dnn >>> cnn (3차)
# 걸린시간 : 47.97 초
# loss : 0.053280770778656006
# acc : 0.9888888597488403
# acc_score : 0.9888888888888889

