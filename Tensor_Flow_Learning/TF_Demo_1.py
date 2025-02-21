# Linear Regression

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# x =5
# y =6

#print(tf.add(x,y))

tf.compat.v1.disable_eager_execution()

a = tf.constant(5)
b = tf.constant(2)
c = a+b


with tf.compat.v1.Session() as sess:
    result = sess.run(c)
    print(result)
