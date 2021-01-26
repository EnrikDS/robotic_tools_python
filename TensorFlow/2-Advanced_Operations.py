#gradient() performs gradient operations
#Optimum where gradient = 0
import tensorflow as tf
x = tf.Variable(-1.0)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.multiply(x, x)

g = tape.gradient(y, x)
print(g.numpy())

#Reshaping tensors
gray_tensor = tf.constant(2, shape=[28,28])
gray_vector = tf.reshape(gray_tensor, (28*28, 1))
print(gray_vector)

def compute_gradient(x0):
    # Define x as a variable with an initial value of x0
    x = tf.Variable(x0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        # Define y using the multiply operation
        y = tf.multiply(x, x)
    # Return the gradient of y with respect to x
    return tape.gradient(y, x).numpy()

# Compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))
