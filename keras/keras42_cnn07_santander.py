# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

# 22 카피


import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

path = 'c:/study/_data/kaggle_santander/' #학원
# path = 'D:\\Furiosa_AI\\study_3주차\\_data\\kaggle_santander\\' #집

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

exit

########### X 데이터를 CNN에 넣기 위해 4차원으로 변환
x_train = x_train.reshape(-1,20,10,1)
x_test = x_test.reshape(-1,20,10,1)

print(x_train.shape, x_test.shape) # (140000, 20, 10, 1) (60000, 20, 10, 1)
print(y_train.shape, y_test.shape) # (140000,) (60000,)

# exit()

#2. 모델구성
model = Sequential()

model.add(Conv2D(64, (2,2), input_shape=(20, 10, 1))) 
model.add(Conv2D(32, (3,3) ,activation='relu' )) 
model.add(Conv2D(32, (2,2) ,padding='same', activation='relu' )) 
model.add(MaxPooling2D())

# model.add(Flatten())
model.add(GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))


model.add(Dense(1, activation='sigmoid'))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['acc'],

              )

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    patience=50,
)

start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=1000,
    verbose=1,
    validation_split=0.3,
    callbacks=[es],
)
end_time = time.time()

#4. 성능, 평가
loss = model.evaluate(x_test, y_test)
print("loss : ", round(loss[0],4))
print("acc : ", round(loss[1],4))

y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score : ", round(acc_score,4))
print("걸린시간 : ", round((end_time - start_time),2),"초") 


################################## RobustScaler ##################### ==> 성능하향
# Epoch 56/100
# 걸린시간 :  13.8 초
# loss :  0.244
# acc :  0.9092
# acc_score :  0.9092

########### 드랍아웃 ==> 하향
# loss :  0.2626
# acc :  0.8995
# acc_score :  0.8995

######### dnn >>> cnn 1차
# loss :  0.2161
# acc :  0.9217
# acc_score :  0.9217
# 걸린시간 :  316.91 초