import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
# path = "C:\\study\\_data\\kaggle_bike" # or ./_data/kaggle_bike/

path = "./_data/kaggle_bike/"

# train_csv = pd.read_csv(path + "train.csv", index_col=0)
# test_csv = pd.read_csv(path + "test.csv", index_col=0)
# submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

train_csv = pd.read_csv("C:\\study\\_data\\kaggle_bike\\train.csv")
test_csv = pd.read_csv("C:\\study\\_data\\kaggle_bike\\test.csv")
submission = pd.read_csv("C:\\study\\_data\\kaggle_bike\\sampleSubmission.csv")

# 데이터 확인
print("train_csv : ",train_csv) # [10886 rows x 11 columns]
print("test_csv : ",test_csv) # [6493 rows x 8 columns]
print("submission : ", submission) # [6493 rows x 1 columns]