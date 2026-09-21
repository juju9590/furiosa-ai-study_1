# 19-2 copy

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Dropout,Input
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

from tensorflow.keras.callbacks import EarlyStopping


# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442, )

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                train_size=0.8, 
                                random_state=333,
                                
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


# 2. 모델 구성
# model = Sequential()
# model.add(Dense(30, input_dim=10, activation='relu'))
# model.add(Dropout(0.2)) #하이퍼파라미터
# model.add(Dense(60, activation='relu'))
# model.add(Dense(80, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(40, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1))

################# 함수형 모델
input1 = Input(shape=(10,))                                         # 입력층
dense1 = Dense(30, activation='relu', name='layer_1')(input1)        
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(60, activation='relu', name='layer_2')(drop1)        
dense3 = Dense(80, name='layer_3')(dense2)                           
drop2 = Dropout(0.5)(dense3)
dense4 = Dense(40, activation='relu', name='layer_4')(drop2)  
drop3 = Dropout(0.3)(dense4)
output1 = Dense(1)(drop3)

model = Model(inputs=input1, outputs=output1)                       # 모델정의


# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")

import time
start_time = time.time()

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    restore_best_weights=True,
    patience=50,

)

hist = model.fit(x_train, y_train, 
          epochs=500, 
          batch_size=64,
          validation_split=0.20,
          callbacks = [es],
          )

end_time = time.time()
print("걸린시간 :", round(end_time-start_time,2), "초")

print("================== 학습 종료 ======================")

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", rmse)


################### RobustScaler ############ =============> 성능향상

# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 3169.5786 
# loss :  3169.57861328125
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
# r2 :  0.40218118384415724
# rmse :  56.29900937746963

############ 드랍아웃 적용
# ================== 학습 종료 ====================== 향상
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 2897.8811 
# loss :  2897.881103515625
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
# r2 :  0.4534264267324164
# rmse :  53.8319705855693
# PS C:\study> 


################# 함수형 모델
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 2965.2178 
# loss :  2965.2177734375
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
# r2 :  0.44072600458401245
# rmse :  54.45381057743559