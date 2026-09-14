# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

# 22 카피


import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

path = 'c:/study/_data/kaggle_santander/' #절대경로

train_csv = pd.read_csv(path + "train.csv", index_col=0) #파일이 있는 경로 표시
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

print(train_csv.shape) #(200000, 201)
print(test_csv.shape) #(200000, 200)
print(submission_csv.shape) #(200000, 1)

# 결측치 보기

# print(train_csv.info()) # 1
print(train_csv.isna().sum()) #2
print(test_csv.isnull().sum()) #3

# 데이터
x = train_csv.drop(['target'], axis=1) 
y = train_csv['target']
print(x.shape, y.shape) #(200000, 200) (200000,)
print(np.unique(y, return_counts=True)) #(array([0, 1]), array([179902,  20098]))

x_train, x_test, y_train, y_test = train_test_split(x, y,
                 train_size=0.7,
                 random_state=345,
                 stratify=y,

                 )

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()


scaler.fit(x_train) 

x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)    

print(np.min(x_train), np.max(x_train)) 
print(np.min(x_test), np.max(x_test))

print(x_train.shape, y_train.shape) #(140000, 200) (140000,)

#2. 모델구성
path ='./_save/keras31/'
model = load_model(path + 'k31_07_0914_1451-0005-0.2482.keras')



# model = Sequential()
# model.add(Dense(30, input_dim=200, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(80, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# es = EarlyStopping(
#     monitor='val_loss',
#     mode='auto',
#     restore_best_weights=True,
#     patience=20,
#     verbose=1,
# )

# import datetime
# date = datetime.datetime.now() 
# print(date) 
# print(type(date)) 
# date = date.strftime("%m%d_%H%M") 
# print(date) #
# print(type(date)) 

# path ='./_save/keras31/'

# filename = '{epoch:04d}-{val_loss:.4f}.keras'
# filepath = "".join([path, "k31_07_", date, "-" ,filename])

 
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only=True,
#     filepath=filepath,
#     verbose=1,
# )

# #3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', 
#               optimizer='adam',
#               metrics=['acc'],

#               )


# start_time = time.time()
# model.fit(
#     x_train, y_train,
#     epochs=100,
#     batch_size=1000,
#     verbose=1,
#     validation_split=0.3,
#     callbacks=[es,mcp],
# )
# end_time = time.time()
# print("걸린시간 : ", round((end_time - start_time),2),"초") 

# print("================================= 학습 종료 =================================")

#4. 성능, 평가
loss = model.evaluate(x_test, y_test)
print("loss : ", round(loss[0],4))
print("acc : ", round(loss[1],4))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score : ", round(acc_score,4))

############ 결과 save
# loss :  0.2461
# acc :  0.9085

# acc_score :  0.9084

############ 결과 load
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 602us/step - acc: 0.9085 - loss: 0.2461  
# loss :  0.2461
# acc :  0.9085
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 345us/step 
# acc_score :  0.9084



