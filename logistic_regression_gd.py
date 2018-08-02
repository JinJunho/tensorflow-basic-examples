#!/usr/bin/python

from __future__ import print_function
import tensorflow as tf
import numpy as np

data = np.loadtxt("trainDatasetForLogisticRegression", unpack=True, dtype="float32")

x_data = data[:-1]
y_data = data[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

h = tf.matmul(W, X)
H = tf.div(1., 1. + tf.exp(-h))

cost = -tf.reduce_mean(Y * tf.log(H) + (1 - Y) * tf.log(1 - H))

optimizer = tf.train.GradientDescentOptimizer(0.005)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

session = tf.Session()
session.run(init)

for step in range(5001):
	session.run(train, feed_dict={X: x_data, Y: y_data})
	if step % 500 == 0:
		print(step, session.run(cost, feed_dict={X: x_data, Y: y_data}), session.run(W))

print("---------------------------------------------")

print("[1,2,2]:", session.run(H, feed_dict={X:[[1], [2], [2]]}))
print("[1,6,6]:", session.run(H, feed_dict={X:[[1], [6], [6]]}))
print(session.run(H, feed_dict={X:[[1,1],[2,6],[2,6]]}) > 0.5)

session.close()
