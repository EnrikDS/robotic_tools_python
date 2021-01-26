''' Introuduction to Tensor Flow
'''
import tensorflow as tf
# 0D Tensor
d0 = tf.ones((1,))

#1D Tensor
d1 = tf.ones((2,))

#2D Tensor
d2 = tf.ones((2, 2))

#3D Tensor
d3 = tf.ones((2, 2, 2))

print(d0, d1, d2, d3)

#Printing transforming to numpy
print(d0.numpy())
print(d1.numpy())
print(d2.numpy())
print(d3.numpy())

#Constants
from tensorflow import constant
a = constant(3, shape=[2,3])
b = constant([1, 2, 3, 4], shape = [2,2])

#Variables
a0 = tf.Variable([1, 2, 3, 4, 5, 6], dtype=tf.float32)
b = tf.constant(2, tf.float32)
c0 = tf.multiply(a0, b)
c1 = a0*b
print(c0.numpy(), c1.numpy())

# The add() operation performs element-wise addition
# Element wise multiplication is done with multiply()
# Matrix multiplication is done with matmul()
# reduce_sum() sums over the diemnsions of a tensor


