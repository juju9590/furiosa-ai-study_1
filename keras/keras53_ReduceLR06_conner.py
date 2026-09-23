# 분류
# AI 모델은 분류(이진,다중 분류)와 회귀 모델 2가지만 있다.

# 21 카피

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #(필수)이진분류모델


#3.컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
# learning_rate = 0.001 # 디폴트 
learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='binary_crossentropy', #(필수)이진분류모델
              optimizer=Adam(learning_rate=learning_rate), 
              metrics=['accuracy'], 
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
model.fit(x_train, y_train,
          epochs=100,
          batch_size=32,
          validation_split=0.3, #트레인에서 검증부분 분리
          verbose=1,
          callbacks=[es, rlr],
          )
end_time = time.time()
print("걸린시간 :", round(end_time-start_time,2), "초")


#4. 성능, 평가
loss = model.evaluate(x_test, y_test)
print("loss : ", round(loss[0],4)) # loss=binary_crossentropy
print("accuracy : ", round(loss[1],4))


# loss는 상대적인 성질을 가지고 있어 1번만 훈련해서 나온 loss값은 잘했다고 할 수없다.
# 하지만 분류모델은 보조지표를 가지고 0,1을 정확하게 맞추는지 정확도를 알아볼 수 있다. 보조지표는 컴파일에서 적용

# print("======== 예측값(시그모이드까지만 적용, 반올림 미적용) 앞에서 10개만 =============")
y_pred = model.predict(x_test)
# print(y_pred[:10]) # 0~1사이의 값을 뽑아준다.. 앞 10개만 뽑아본다면?
# [[0.9978364 ]
#  [0.9925489 ]
#  [0.9972778 ]
#  [0.9913201 ]
#  [0.00185576]
#  [0.857538  ]
#  [0.99470586]
#  [0.9945206 ]
#  [0.9979957 ]
#  [0.02947073]]

y_pred = np.round(y_pred) # 반올림 처리
# print(y_pred[:10]) # 반올림 처리하여 0,1로 추출한다.
# [[1.]
#  [1.]
#  [1.]
#  [1.]
#  [0.]
#  [1.]
#  [1.]
#  [1.]
#  [1.]
#  [0.]]

# 이진분류 모델 '치트키'
# ㄴ 모델의 마지막에 sigmoid + binary_crossentropy + y_pred 반올림

# 
from sklearn.metrics import accuracy_score 
# acc_score = accuracy_score(y_test, y_pred) 
# y_test = [0, 1], y_pred =[0~1사이값]
# Classification metrics can't handle a mix of binary and continuous targets
# 분류 매트릭스는 이진수와 연속되는 y값은 핸들링 할 수 없다 => y예측값을 반올림 처리한 후 값을 추출한다

acc_score = accuracy_score(y_test, y_pred)  
print("acc_score : ", acc_score ) # acc_score :  0.9122807017543859


# y_pred = model.predict(x_test)

# r2 = r2_score(y_test, y_pred) #r2는 y원값에서 y예측값 빼주기
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# # R² 1에 가까움 → 잘함
# # R² 0 → 평균값 찍는 정도
# # R² 음수 → 평균값 찍는 것보다 못함

# print("r2_score : ", r2)
# print("rmse : ", rmse)

# 회귀 → 숫자 예측 → R², MSE, RMSE
# 분류 → 종류 예측 → Accuracy, Precision, Recall, F1


# ========================== MinMaxScaler 적용 후  =========================== ==> 향상
# loss :  0.0681
# accuracy :  0.9883
# acc_score :  0.9883040935672515



# learning_rate = 0.01
# 걸린시간 : 1.97 초
# loss :  0.0977
# accuracy :  0.9766
# acc_score :  0.9766081871345029

# learning_rate = 0.01 Reduce
# 걸린시간 : 4.67 초
# loss :  0.0963
# accuracy :  0.9591
# acc_score :  0.9590643274853801


# learning_rate = 0.0001 Reduce
# 걸린시간 : 7.76 초
# loss :  0.0834
# accuracy :  0.9766
# acc_score :  0.9766081871345029





