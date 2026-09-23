# 28-10 카피

from sklearn.datasets import load_digits

# acc = 1.0

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
# print(datasets.DESCR)
print(x.shape, y.shape) #(1797, 64) (1797,)

# print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

# print(" ================= 원핫1. to_categorical =======================")

# from tensorflow.keras.utils import to_categorical 
# y = to_categorical(y)
# print(y)
# print(y.shape) #(1797, 10)

# print(" ================= 원핫2. pd.get_dummies =======================")

# y = pd.get_dummies(y, dtype=int)
# print(y)

print(" ================= 원핫3.OneHotEncoder =======================")
from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1, 1) # 1차원의 데이터를 2차원의 데이터로 변환
# print(y)

ohe = OneHotEncoder(sparse_output=False) #원핫인코더 정의 + 넘파이 배열로 나타내기 위해 sparse_output=False
y = ohe.fit_transform(y)
# print(y, y.shape)


# exit()

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                    test_size=0.2,
                                    random_state=333,
                                    shuffle=True,
                                    stratify=y,

                                    )

print(x_train.shape, x_test.shape) # (1437, 64) (360, 64)
print(y_train.shape, y_test.shape) # (1437, 10) (360, 10)

# 정규화

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler() # 1번
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()


scaler.fit(x_train) 

x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) 
print(np.min(x_test), np.max(x_test))

#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=64, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
# learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
# learning_rate = 0.005
learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(learning_rate=learning_rate), 
              metrics=['acc'],)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=50,
    verbose=1,
    restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1,
    factor=0.5, # learning_rate(러닝레이트) 비율 조절

)

start_time = time.time()
model.fit(x_train, y_train,
          epochs=500,
          batch_size=4,
          validation_split=0.2,
          callbacks=[es,rlr],
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







###################### MaxAbsScaler  ############## ====> 성능갱신
# 걸린시간 : 15.84 초
# loss : 0.04698524624109268
# acc : 0.9888888597488403
# acc_score : 0.9888888888888889

# 걸린시간 : 14.12 초
# loss : 0.17170606553554535
# acc : 0.9583333134651184
# acc_score : 0.9583333333333334

# learning_rate = 0.005
# loss : 0.10436984151601791
# acc : 0.9777777791023254
# acc_score : 0.9777777777777777

# learning_rate = 0.005, ReduceLR
# 걸린시간 : 22.25 초
# loss : 0.1411261260509491
# acc : 0.9722222089767456
# acc_score : 0.9722222222222222

# learning_rate = 0.05, ReduceLR
# 걸린시간 : 43.05 초
# loss : 2.3058958053588867
# acc : 0.10000000149011612
# acc_score : 0.1
