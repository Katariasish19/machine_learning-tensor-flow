import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

a = tf.constant(6)
b = tf.constant(9)
x = tf.Variable(0)


r1= tf.add(a,b)
r2 =tf.subtract(a,b)
r3 = tf.multiply(a,b)
r4 = tf.divide(a,b)
r5 = tf.less(a,b)
r6 =tf.greater(a,b)
r7 =tf.equal(a,b)
r8 = tf.cond(a>b, lambda: tf.add(a,b), lambda:tf.subtract(a,b))

increment = tf.constant(2)

for i in range(18):
    x.assign_add(increment)



tf.print(r1)
tf.print(r2)
tf.print(r3)
tf.print(r4)
tf.print(r5)
tf.print(r6)
tf.print(r8)
tf.print(x)

