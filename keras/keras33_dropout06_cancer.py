# 분류
# AI 모델은 분류(이진,다중 분류)와 회귀 모델 2가지만 있다.

# 21 카피

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.metrics import r2_score, mean_squared_error (회귀모델)

from sklearn.datasets import load_breast_cancer #데이터셋 불러오기(유방암관련)

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR) #실무에서는 사용할 일 없다. 이 데이터는 사이킷런에서 제공하는 데이터일 뿐 (캐글이나 데이콘에서 제공하는 데이터는 실무에 사용)
print(datasets.feature_names) # 컬럼 30개의 이름 보여줌, 실무에서는 판다스로 확인 (학습용)

# x = datasets.data
x = datasets['data'] # datasets는 딕셔너리(key, value) 형태의 데이터로 키값을 리스트로 해 놓으면 벨률를 불러올 수 있다.
y = datasets.target # ['target']

print(x.shape, y.shape) #(569, 30) (569,)
print('타입 : ', type(x)) # x 데이터의 타입을 확인 = <class 'numpy.ndarray'>
# 가장 많이 발생하는 오류 : 오타, 경로, 버전 ...

print(y) # 분류형 모델을 할 때는 y를 꼭 확인해야 함 (y=범주) [0,1,0,0,...,1,1,1,0,1]

# 0과 1의 개수가 몇개인지 찾아보기 -numpy
print("y의 범주 : ",np.unique(y)) # [0 1] 
# 분류형일경우 범주의 크기가 비슷하면 성능이 더 좋다. 범주형은 반드시 y의 범주 갯수를 확인해야 한다 
print("y의 범주의 갯수 : ",np.unique(y, return_counts=True)) 
# (array([0, 1]), array([212, 357])) => 0은 212개, 1은 357개 

# 0과 1의 개수가 몇개인지 찾아보기 -pandas
print(pd.DataFrame(y).value_counts()) # 판다스에서는 데이터프레임과 시리즈 2가지만 있다.
# 1    357
# 0    212
print(pd.Series(y).value_counts())
# 1    357
# 0    212
# 판다스든 넘파이든 사용하고 싶은거 쓰면 됨

# 열=컬럼=피처(feature)=속성=특성=Attributes

# 데이터셋 트레인,테스트 분리(7:3)
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    random_state=908,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    stratify=y, # (중요) y데이터를 스트레이트파이하란 이야기
                                                    # stratify = '층을 이루게 하다', '계층화하다', '계층별로 나누다'
                                                    )

#  1) stratify=y 미설정 시 
print(np.unique(y_train, return_counts=True))
# (array([0, 1]), array([150, 248]))
print(np.unique(y_test, return_counts=True))
# (array([0, 1]), array([ 62, 109])) ==> 학습에 영향을 미치지 않기 때문에 0,1 차이가 나도 문제 없다.

# 중요 2) stratify=y 설정시
print("==== y_train의 종류와 종류별 트레인(70%)로 분리 ====")
print(np.unique(y_train, return_counts=True))
# (array([0, 1]), array([148, 250]))
print("==== y_test의 종류와 종류별 테스트(30%)로 분리 ====")
print(np.unique(y_test, return_counts=True))
# (array([0, 1]), array([ 64, 107])) ==> y를 기준으로 7:3 비율로 분리 해준다

print(x_train.shape, x_test.shape) # (398, 30) (171, 30)
print(y_train.shape, y_test.shape) # (398,) (171,)




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





#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=30, activation='relu'))

model.add(Dense(40, activation='relu'))
model.add(Dropout(0.5))


model.add(Dense(40, activation='relu'))
model.add(Dropout(0.3))


model.add(Dense(40, activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(40, activation='relu'))
model.add(Dropout(0.5))


model.add(Dense(40, activation='relu'))

model.add(Dense(1, activation='sigmoid')) #(필수)이진분류모델


#3.컴파일, 훈련
model.compile(loss='binary_crossentropy', #(필수)이진분류모델
              optimizer='adam', # 아담이 그라디언트, 가중치 갱신 계산
              metrics=['accuracy'], 
              ) 


es = EarlyStopping(
    monitor="val_loss",
    mode='min',
    patience=20,
    restore_best_weights=True
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs=100,
          batch_size=32,
          validation_split=0.3, #트레인에서 검증부분 분리
          verbose=1,
          callbacks=[es],
          )
end_time = time.time()
print("걸린시간 :", round(end_time-start_time,2), "초")

print("===================== 학습완료 ===========================")

#4. 성능, 평가
print("=========================================================")
loss = model.evaluate(x_test, y_test)
print("loss : ", round(loss[0],4)) # loss=binary_crossentropy
print("accuracy : ", round(loss[1],4))
print("=========================================================")



y_pred = model.predict(x_test)
y_pred = np.round(y_pred) # 반올림 처리

from sklearn.metrics import accuracy_score 

acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", acc_score ) # acc_score :  0.9122807017543859


# ========================== 결과 ===========================

# Epoch 45/100
# 9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9353 - loss: 0.1656 - val_accuracy: 0.9167 - val_loss: 0.2040
# ===================== 학습완료 ===========================
# ======================================================
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9181 - loss: 0.1908 
# loss :  0.1908
# accuracy :  0.9181
# ======================================================
# ======== 예측값(시그모이드까지만 적용, 반올림 미적용) 앞에서 10개만 =============
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 
# acc_score :  0.9181286549707602
# PS C:\study> 


# ========================== MinMaxScaler 적용 후  =========================== ==> 향상
# Epoch 100/100
# 9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9964 - loss: 0.0092 - val_accuracy: 0.9917 - val_loss: 0.0668
# ===================== 학습완료 ===========================
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9883 - loss: 0.0681 
# loss :  0.0681
# accuracy :  0.9883
# ======== 예측값(시그모이드까지만 적용, 반올림 미적용) 앞에서 10개만 =============
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
# acc_score :  0.9883040935672515

# ############## StandardScaler 적용 후  ################### ==> 향상
# Epoch 29/100
# 9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 1.0000 - loss: 0.0012 - val_accuracy: 0.9667 - val_loss: 0.1402
# 걸린시간 : 2.73 초
# ===================== 학습완료 ===========================
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9591 - loss: 0.0872 
# loss :  0.0872
# accuracy :  0.9591
# =========================================================
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
# acc_score :  0.9590643274853801

# ############## MaxAbsScaler 적용 후  ################### ==> loss 하향, 정확도 향상
# Epoch 100/100
# 9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.9784 - loss: 0.0592 - val_accuracy: 0.9917 - val_loss: 0.0423
# 걸린시간 : 7.14 초
# ===================== 학습완료 ===========================
# =========================================================
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9532 - loss: 0.1173 
# loss :  0.1173
# accuracy :  0.9532
# =========================================================
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
# acc_score :  0.9532163742690059

# ############## RobustScaler 적용 후  ################### ==> loss 하향, 정확도 향상
# Epoch 29/100
# 9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 1.0000 - loss: 0.0023 - val_accuracy: 0.9417 - val_loss: 0.1868
# 걸린시간 : 2.86 초
# ===================== 학습완료 ===========================
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9649 - loss: 0.1388 
# loss :  0.1388
# accuracy :  0.9649
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 
# acc_score :  0.9649122807017544

############ 드롭아웃 ==> loss 향상
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.9649 - loss: 0.0864 
# loss :  0.0864
# accuracy :  0.9649
# =========================================================
# 6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
# acc_score :  0.9649122807017544




