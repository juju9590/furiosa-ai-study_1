# 23-4 카피

from sklearn.datasets import load_digits

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time, datetime
from sklearn.metrics import accuracy_score

path = './_save/keras31'

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
# print(datasets.DESCR)
print(x.shape, y.shape) #(1797, 64) (1797,)

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


# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# # scaler = MinMaxScaler() # 1번
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# scaler = RobustScaler()


# scaler.fit(x_train) 

# x_train = scaler.transform(x_train)  
# x_test = scaler.transform(x_test)    

# print(np.min(x_train), np.max(x_train)) 
# print(np.min(x_test), np.max(x_test))

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

# es = EarlyStopping(
#     monitor='val_loss',
#     mode='auto',
#     patience=20,
#     restore_best_weights=True,
#     verbose=1,
# )

# date = datetime.datetime.now()
# date = date.strftime('%d%m-%H%M')

# path ='./_save/keras31/'

# filename = '{epoch:04d}-{val_loss:.4f}.keras'
# filepath = "".join([path, "k31_10_", date, "-", filename])

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only=True,
#     filepath=filepath,
#     verbose=1,
# )

start_time = time.time()
model.fit(x_train, y_train,
          epochs=100,
          batch_size=4,
          validation_split=0.2,
        #   callbacks=[es,mcp],
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


########### 결과 save
# loss : 0.14555810391902924
# acc : 0.9666666388511658
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
# acc_score : 0.9666666666666667

# epochs=100
########### GPU
# 걸린시간 :  

########### CPU 
# 걸린시간 : 30.87 초




