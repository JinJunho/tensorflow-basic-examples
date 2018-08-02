#!/usr/bin/python

import tensorflow as tf
import matplotlib.pyplot as plt

x = [1., 2., 3.]
y = [1., 2., 3.]
m = len(x)

W = tf.placeholder(tf.float32)

hypothesis = tf.mul(W, x)
cost = tf.reduce_sum(tf.pow(hypothesis-y, 2))

init = tf.initialize_all_variables()

session = tf.Session()
session.run(init)

wv, cv = [], []

for i in range(-30, 50):
	xPos = i*0.1
	yPos = session.run(cost, feed_dict={W: xPos})
	print("{:3.1f}, {:3.1f}".format(xPos, yPos))

	wv.append(xPos)
	cv.append(yPos)

session.close()

plt.plot(wv, cv, "ro")
plt.ylabel("cost")
plt.xlabel("W")
plt.show()

