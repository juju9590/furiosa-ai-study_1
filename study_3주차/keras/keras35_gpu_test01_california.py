# 30-1 카피

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import time

# path ='./_save/keras30/'

#1. 데이터
datasets = fetch_california_housing() 
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(20640, 8) (20640,)


x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.8,
    # test_size=0.2,
    # shuffle=True, #디폴트 섞는다
    random_state=777,
) 

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# # scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# scaler = RobustScaler() # 이상치에 강력

# # x_train = scaler.transform(x_train)  # <===  x_train 스케일링 시행
# # x_test = scaler.transform(x_test)    # <===  x_test 스케일링 시행

# x_train = scaler.fit_transform(x_train) # 한줄로 사욯 가능
# x_test = scaler.transform(x_test)    


# print(np.min(x_train), np.max(x_train)) 
# print(np.min(x_test), np.max(x_test)) 

#2. 모델구성
model = Sequential()
model.add(Dense(2, activation='relu', input_dim=8))
model.add(Dense(6, activation='relu'))
model.add(Dense(12))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))


# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

# es = EarlyStopping(
#     monitor='val_loss',
#     mode='min',
#     patience=20,
#     restore_best_weights=True,
#     verbose=1, #훈련중에 es가 진행되는지 볼 수 있음
# )

####### mcp save 파일명 만들기 (일반적일 파일명도 동일) ########### 시작
# import datetime
# date = datetime.datetime.now() # 현재 시간 반환
# print(date) #2026-09-14 11:42:10.818328
# print(type(date)) # <class 'datetime.datetime'>
# date = date.strftime("%m%d_%H%M") #string for time 함수로 문자로 준비/ 소문자, 대문자 구분 
# print(date) # 0914_1147
# print(type(date)) # <class 'str'>

# path ='./_save/keras30/'
# filename = '{epoch:04d}-{val_loss:.4f}.keras'
# # d : int 형태, f : float 형태
# filepath = "".join([path, "k30_", date, "-" ,filename])

# # 파일명 조합 예시
# # './_save/keras30/' + 'k30_' + '0914_1147' + '0530-0.001 .keras'
# 파일명 : k30_0914_1323-0064-0.3791.keras

####### mcp save 파일명 만들기 (일반적일 파일명도 동일) ########### 끝


# exit()

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only=True, 
#     filepath=filepath, 
#     verbose=1, 
# )

start_time = time.time()
hist = model.fit(x_train, 
                 y_train, 
                epochs=100, 
                batch_size=32,
                validation_split=0.2,
                # callbacks = [es, mcp],
                verbose=1,                              
                ) 

end_time = time.time() # 현재시간을 반환, = 끝시간
print("걸린시간 :", round(end_time-start_time,2), "초")

print("=================== 학습 종료 ========================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, )
print("loss :", loss)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 : ", round(r2,3))

mse = mean_squared_error(y_test, y_pred)
print("mse : ", round(mse,3))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse : ", round(rmse,3))



######################## 결과
# r2 :  0.72
# mse :  0.361
# rmse :  0.601

# epochs : 100, es, mcp 제외
############## GPU 속도
# 걸린시간 : 68.2 초

############## CPU 속도 ㅇ
# 걸린시간 : 40.02 초


