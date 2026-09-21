import numpy as np

x1 = np.array([1,2,3]) # (3,) 벡터
# 데이터를 받으면 가장 먼저 할 일 : 구조를 확인하기 (쉐이프)
print(" x1 = ", x1.shape)
# 실행 결과 값 : x1 =  (3,)

x2 = np.array([[1,2,3]]) #(1,3) 행렬
print(" x2 = ", x2.shape)
# 실행결과값  x2 =  (1, 3)

x3 = np.array([[1,2],[3,4]]) #(2,2)
print(" x3 = ", x3.shape)
# 실해결과값 : x3 =  (2, 2)

x4 = np.array([[1,2],[3,4],[5,6]]) # x4 =  (3, 2)
print(" x4 = ", x4.shape)
# 뒷자리는 백터의 갯수로 맞춰야 한다.

# x4_1 = np.array([[1,2],[3,4],[5,6,7]]) 
# print(" x4_1 = ", x4_1.shape)

# File "c:\study\keras\keras07_행렬.py", line 20, in <module>
#     x4_1 = np.array([[1,2],[3,4],[5,6,7]])
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

x5 = np.array([[[1,2],[3,4],[5,6]]]) # x5 =  (1, 3, 2)
print(" x5 = ", x5.shape)

x6 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]]) # x6 =  (2, 2, 2)
print(" x6 = ", x6.shape)


x7 = np.array([[[[[1,2,3,4,5],[6,7,8,9,10]]]]]) #  x7 =  (1, 1, 1, 2, 5)
print(" x7 = ", x7.shape)

x8 = np.array([[[1,2,3]],[[4,5,6]]]) # x8 =  (2, 1, 3)
print(" x8 = ", x8.shape)

x9 = np.array([[[[1]]],[[[2]]]]) # x9 =  (2, 1, 1, 1)
print(" x9 = ", x9.shape)

