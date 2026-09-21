# 가상환경 tf29x-gpu 에서 실행

import tensorflow as tf
print(tf.__version__) # 2.9.3

gpus = tf.config.experimental.list_physical_devices('GPU') 
# gpus = tf.config.list_physical_devices('GPU')
# 이 부분은 컴퓨터에 있는 GPU 중에서 TensorFlow가 실제로 사용할 수 있는 GPU를 찾는 코드
print(gpus)
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

if(gpus):
    print('GPU 돈다!!')
else :
    print('GPU 없다!!')
# GPU 돈다!!


# 가상환경(py311)에서 실행 => 아래와같이 나오는 이유는 tf 버전이 맞지 않아 발생.
# 2.21.0
# []            
# GPU 없다!! 





