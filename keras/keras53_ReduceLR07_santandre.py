# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

# 22 카피


import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

path = 'c:/study/_data/kaggle_santander/' #절대경로
# path = './_data/kaggle_santander/' #상대경로

train_csv = pd.read_csv(path + "train.csv", index_col=0) #파일이 있는 경로 표시
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

# print(train_csv.shape) #(200000, 201)
# print(test_csv.shape) #(200000, 200)
# print(submission_csv.shape) #(200000, 1)

# 결측치 보기

# print(train_csv.info()) # 1
# print(train_csv.isna().sum()) #2
# print(test_csv.isnull().sum()) #3

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

# print(np.min(x_train), np.max(x_train)) 
# print(np.min(x_test), np.max(x_test))

print(x_train.shape, y_train.shape) #(140000, 200) (140000,)

#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=200, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
# learning_rate = 0.001 # 디폴트 
# learning_rate = 0.0001
learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='binary_crossentropy', 
              optimizer=Adam(learning_rate=learning_rate),
              metrics=['acc'],
              )

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
model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=1000,
    verbose=1,
    validation_split=0.3,
    callbacks=[es, rlr,],
)
end_time = time.time()
print("걸린시간 : ", round((end_time - start_time),2),"초") 

print("================================= 학습 종료 =================================")

#4. 성능, 평가
loss = model.evaluate(x_test, y_test)
print("loss : ", round(loss[0],4))
print("acc : ", round(loss[1],4))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score : ", round(acc_score,4))

#1차 시도
# loss :  0.244
# acc :  0.9105
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 337us/step 
# acc_score :  0.9104666666666666

# ########### submission.csv 만들기 // target 컬럼에 넣어준다 #############
# # print(submission_csv)
# y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)

# submission_csv['target'] = y_submit
# print(submission_csv)

# submission_csv.to_csv(path + "submit/" + "submit_0908_1737.csv")


####### MaxAbsScaler 
# 걸린시간 :  16.79 초
# loss :  0.2414
# acc :  0.9107
# acc_score :  0.9106


# learning_rate = 0.0001
# 걸린시간 :  23.55 초
# loss :  0.247
# acc :  0.9086
# acc_score :  0.9086


# # learning_rate = 0.0001, ReduceLR
# 걸린시간 :  25.31 초
# loss :  0.247
# acc :  0.9086
# acc_score :  0.9086

# # learning_rate = 0.005, ReduceLR
# 걸린시간 :  18.33 초
# loss :  0.2439
# acc :  0.9093
# acc_score :  0.9093