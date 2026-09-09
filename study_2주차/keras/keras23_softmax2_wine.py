from sklearn.datasets import load_wine

## acc = 0.95 이상

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_wine()
# print(datasets.DESCR)
'''
**Data Set Characteristics:**

:Number of Instances: 178
:Number of Attributes: 13 numeric, predictive attributes and the class
:Attribute Information:
    - Alcohol
    - Malic acid
    - Ash
    - Alcalinity of ash
    - Magnesium
    - Total phenols
    - Flavanoids
    - Nonflavanoid phenols
    - Proanthocyanins
    - Color intensity
    - Hue
    - OD280/OD315 of diluted wines
    - Proline
    - class:
        - class_0
        - class_1
        - class_2
'''
x = datasets['data']
y = datasets['target']
print(x.shape, y.shape) #(178, 13) (178,)

############ 원핫인코딩 ###############
# print(" =========== 원핫인코딩 to_categorical =============== ")
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# # print(y)
# '''
# [0. 0. 1.]
#  [0. 0. 1.]
#  [0. 0. 1.]
#  [0. 0. 1.]
#  '''
# print(y.shape) #(178, 3)

print(" =========== 원핫인코딩  pd.get_dummies(pd.DataFrame()) =============== ")

df = pd.DataFrame(y)
print(df)










exit()

print("================ train_test_split ================== ")

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=999,
                                                    shuffle=True,
                                                    stratify=y,
                                                    )
print(x_train.shape, x_test.shape) # (124, 13) (54, 13)
print(y_train.shape, y_test.shape) # (124, 3) (54, 3)

# 2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(260, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(3, activation='softmax')) # 다중분류

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'],)

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=500,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(x_train, y_train,
          epochs=5000, batch_size=16,
          verbose=1,
          validation_split=0.2,
          callbacks=[es],
          )

end_time = time.time()
print("걸린시간 :", round((end_time-start_time),2))

print("===================== 학습 끝 ==========================")
# 4. 예측, 평가
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc :", result[1])

y_pred = np.argmax(model.predict(x_test), axis=1)
y_test = np.argmax(y_test, axis=1)
print(y_pred [:10])
print(y_test [:10])

acc_score = accuracy_score(y_test, y_pred)
print("acc_score : ", acc_score)


# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - acc: 0.9444 - loss: 0.2345 
# loss : 0.23451410233974457
# acc : 0.9444444179534912
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 47ms/step
# [2 2 1 1 0 1 0 1 0 2]
# [2 2 1 1 0 1 0 1 0 2]
# acc_score :  0.9444444444444444




