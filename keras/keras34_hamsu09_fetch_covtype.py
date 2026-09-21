# 23-3 카피

from sklearn.datasets import fetch_covtype # 시간체크, 배치사이즈 크게 하기

# acc = 0.93 이상

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Dropout , Input
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = fetch_covtype()
# print(datasets.DESCR)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(581012, 54) (581012,)

print(np.unique(y, return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

print("================= 원핫2. pd.get_dummies()  =====================")

y = pd.get_dummies(y, dtype=int)
print(y, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True,
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



# 2. 모델구성
# model = Sequential()
# model.add(Dense(10, input_dim=54, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(7, activation='softmax'))

########## 함수형 모델 
input1 = Input(shape=(54,))
dense1 = Dense(10, activation='relu')(input1)
dense2 = Dense(10, activation='relu')(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(10, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense3)
dense4 = Dense(10, activation='relu')(drop2)
drop3 = Dropout(0.5)(dense4)
dense5 = Dense(10, activation='relu')(drop3)
output1 = Dense(7)(dense5)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'],)

es = EarlyStopping(
                   monitor='val_loss',
                   mode='auto',
                   restore_best_weights=True,
                   patience=20000,
                   )

start_time=time.time()
model.fit(x_train, y_train,
          epochs=10,
          batch_size=128,
          verbose=1,
          validation_split=0.2,
          callbacks=[es],
          )
end_time=time.time()
print("걸린시간 :", round(end_time-start_time,3),"초")
print("======================== 학습종료 =======================")


#4. 예측, 평가
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc :", result[1])

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred )
print("acc_score :", acc_score)

################ 결과 #############################
# Epoch 10/10
# 2906/2906 ━━━━━━━━━━━━━━━━━━━━ 3s 1ms/step - acc: 0.7130 - loss: 0.6783 - val_acc: 0.7248 - val_loss: 0.6640
# 걸린시간 : 29.428 초
# ======================== 학습종료 =======================
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 666us/step - acc: 0.7233 - loss: 0.6642
# loss : 0.6642236709594727
# acc : 0.7233117818832397
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 426us/step 
# acc_score : 0.7233117905733931

###################### MinMaxScaler 적용 후 ################## =======> loss 향상 acc 향상

# 2906/2906 ━━━━━━━━━━━━━━━━━━━━ 3s 968us/step - acc: 0.7484 - loss: 0.5946 - val_acc: 0.7405 - val_loss: 0.6067
# 걸린시간 : 31.511 초
# ======================== 학습종료 =======================
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 674us/step - acc: 0.7479 - loss: 0.5952
# loss : 0.5952174663543701
# acc : 0.7478550672531128
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 428us/step 
# acc_score : 0.747855046771598


###################### StandardScaler 적용 후 ################## =======> 향상

# Epoch 10/10
# 2906/2906 ━━━━━━━━━━━━━━━━━━━━ 3s 929us/step - acc: 0.7606 - loss: 0.5597 - val_acc: 0.7608 - val_loss: 0.5611
# 걸린시간 : 28.665 초
# ======================== 학습종료 =======================
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 668us/step - acc: 0.7617 - loss: 0.5610
# loss : 0.5609884858131409
# acc : 0.7616584897041321
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 424us/step 
# acc_score : 0.7616584769756375


###################### MaxAbsScaler 적용 후 ################## =======> 성능 하향

# 2906/2906 ━━━━━━━━━━━━━━━━━━━━ 3s 924us/step - acc: 0.7443 - loss: 0.5973 - val_acc: 0.7446 - val_loss: 0.5952
# 걸린시간 : 28.52 초
# ======================== 학습종료 =======================
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 660us/step - acc: 0.7441 - loss: 0.5971 
# loss : 0.5971131920814514
# acc : 0.7441374063491821
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 425us/step 
# acc_score : 0.7441374146966946

###################### RobustScaler 적용 후 ################## =======> 성능 개선

# Epoch 10/10
# 2906/2906 ━━━━━━━━━━━━━━━━━━━━ 3s 941us/step - acc: 0.7636 - loss: 0.5473 - val_acc: 0.7661 - val_loss: 0.5466
# 걸린시간 : 28.671 초
# ======================== 학습종료 =======================
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 663us/step - acc: 0.7665 - loss: 0.5480 
# loss : 0.5480256080627441
# acc : 0.7665206789970398
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 412us/step 
# acc_score : 0.7665206578143421

########## 드랍아웃 ===> 하향
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 3s 687us/step - acc: 0.6921 - loss: 0.7334 
# loss : 0.7334348559379578
# acc : 0.6921336054801941
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 421us/step 
# acc_score : 0.6921335937970621

########## 함수형 모델 ==> 하향
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 2s 586us/step - acc: 0.3646 - loss: nan
# loss : nan
# acc : 0.36460331082344055
# 3632/3632 ━━━━━━━━━━━━━━━━━━━━ 1s 312us/step 
# acc_score : 0.3646033234942299