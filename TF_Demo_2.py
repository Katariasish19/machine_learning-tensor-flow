import numpy as np
import tensorflow as tf

# Training data
x_train = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
y_train = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

# Define variables (weights and bias)
w = tf.Variable(0.0)
b = tf.Variable(0.0)

# Define the linear model
def linear_model(x):
    return w * x + b

# Define a loss function (mean squared error)
def loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Define a training loop
learning_rate = 0.01
for i in range(1000):
    with tf.GradientTape() as tape:
        y_pred = linear_model(x_train)
        current_loss = loss(y_train, y_pred)
    
    gradients = tape.gradient(current_loss, [w, b])
    w.assign_sub(gradients[0] * learning_rate)
    b.assign_sub(gradients[1] * learning_rate)

# Print the final values of w and b
tf.print("w:", w)
tf.print("b:", b)
