import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Initialization of tensor
x1 = tf.constant(4, shape=(1,1), dtype=tf.float32)
x2 = tf.constant([[1,2,3], [4,5,6]], dtype=tf.float32)
x3 = tf.ones((3,3), dtype=tf.int32)
x4 = tf.zeros((3,3))
x5 = tf.eye(5)
print(x5)
print("-"*49)

y1 = tf.random.normal((3,3), mean=0, stddev=0.5)
y2 = tf.random.uniform((4,4), minval=-5, maxval=5)
dy = tf.range(start=2, limit=10, delta=0.5)
print(dy)
print("-"*49)

# Mathematical operation
a = tf.constant([1,3,5])
b = tf.constant([2,4,6])

# z = tf.add(a, b) # element z = a + b
# z = tf.subtract(a,b) # z = a - b
# z = tf.divide(a,b) # z = a / b
# z = tf.multiply(a,b) # z = a * b

z = tf.tensordot(a,b, axes = 1) # vector matrix
# z = tf.reduce_sum(a * b, axis=0)
z = a **3
print(z)