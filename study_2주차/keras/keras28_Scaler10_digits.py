# 23-4 카피

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
print(y)

ohe = OneHotEncoder(sparse_output=False) #원핫인코더 정의 + 넘파이 배열로 나타내기 위해 sparse_output=False
y = ohe.fit_transform(y)
print(y, y.shape)


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

######################### 결과 #########################
# Epoch 389/1000
# 144/144 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - acc: 1.0000 - loss: 7.4234e-05 - val_acc: 0.9514 - val_loss: 0.3333
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.9556 - loss: 0.2023 
# loss : 0.20233480632305145
# acc : 0.9555555582046509
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
# acc_score : 0.9555555555555556

###################### MinMaxScaler 적용 후 ############## ====> loss 향상, acc 향상 

# Epoch 591/1000
# 144/144 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - acc: 1.0000 - loss: 3.9647e-05 - val_acc: 0.9583 - val_loss: 0.3389
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - acc: 0.9583 - loss: 0.1980 
# loss : 0.19802841544151306
# acc : 0.9583333134651184
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
# acc_score : 0.9583333333333334

###################### StandardScaler 적용 후 ############## ====> loss, acc 하향

# 288/288 ━━━━━━━━━━━━━━━━━━━━ 0s 973us/step - acc: 1.0000 - loss: 2.8155e-05 - val_acc: 0.9583 - val_loss: 0.1567
# 걸린시간 : 8.53 초
# ================= 훈련 종료 =====================
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - acc: 0.9528 - loss: 0.2903 
# loss : 0.2902926504611969
# acc : 0.9527778029441833
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
# acc_score : 0.9527777777777777

###################### StandardScaler 적용 + 랜덤 바꿈  ############## ====> 성능갱신
# Epoch 26/1000
# 288/288 ━━━━━━━━━━━━━━━━━━━━ 0s 964us/step - acc: 1.0000 - loss: 2.4084e-05 - val_acc: 0.9688 - val_loss: 0.2177
# 걸린시간 : 8.78 초
# ================= 훈련 종료 =====================
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.9833 - loss: 0.0602 
# loss : 0.06017813831567764
# acc : 0.9833333492279053
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
# acc_score : 0.9833333333333333

###################### MaxAbsScaler  ############## ====> 성능갱신
# Epoch 48/1000
# 288/288 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - acc: 1.0000 - loss: 1.1011e-05 - val_acc: 0.9861 - val_loss: 0.0654
# 걸린시간 : 15.84 초
# ================= 훈련 종료 =====================
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.9889 - loss: 0.0470 
# loss : 0.04698524624109268
# acc : 0.9888888597488403
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
# acc_score : 0.9888888888888889

###################### RobustScaler  ############## ====> 성능하향
# Epoch 41/1000
# 288/288 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - acc: 1.0000 - loss: 3.3231e-06 - val_acc: 0.9826 - val_loss: 0.0996
# 걸린시간 : 13.71 초
# ================= 훈련 종료 =====================
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - acc: 0.9583 - loss: 0.1626  
# loss : 0.1625954806804657
# acc : 0.9583333134651184
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
# acc_score : 0.9583333333333334
