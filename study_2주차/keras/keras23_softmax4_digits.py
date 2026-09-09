from sklearn.datasets import load_digits

# acc = 1.0

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
# print(datasets.DESCR)
print(x.shape, y.shape) #(1797, 64) (1797,)

# print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

print(" ================= 원핫1. to_categorical =======================")

# from tensorflow.keras.utils import to_categorical 
# y = to_categorical(y)
# print(y)
# print(y.shape) #(1797, 10)

print(" ================= 원핫2. pd.get_dummies =======================")

# y = pd.get_dummies(y, dtype=int)
# print(y)

print(" ================= 원핫3.OneHotEncoder =======================")
from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1, 1)
print(y)

ohe = OneHotEncoder(sparse_output=False) #원핫인코더 정의
y = ohe.fit_transform(y)
print(y, y.shape)


exit()



x_train, x_test, y_train, y_test = train_test_split(x, y,
                                    test_size=0.2,
                                    random_state=123,
                                    shuffle=True,
                                    stratify=y,

                                    )

print(x_train.shape, x_test.shape) # (1437, 64) (360, 64)
print(y_train.shape, y_test.shape) # (1437, 10) (360, 10)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=64, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'],)

es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=200,
    restore_best_weights=True,
)

model.fit(x_train, y_train,
          epochs=1000,
          batch_size=8,
          validation_split=0.2,
          callbacks=[es],
          )

#4. 예측, 평가
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc :", result[1])

y_pred = np.argmax(model.predict(x_test), axis=1)
y_test = np.argmax(y_test, axis=1)

acc_score = accuracy_score(y_test, y_pred)
print("acc_score :", acc_score)

