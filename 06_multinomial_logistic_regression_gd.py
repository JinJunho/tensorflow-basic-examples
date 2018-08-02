#!/usr/bin/python

from __future__ import print_function
import tensorflow as tf
import numpy as np

data = np.loadtxt("dataset/trainDatasetForSoftmax", unpack=True, dtype="float32")

x_data = np.transpose(data[:3])
y_data = np.transpose(data[3:])

print("x_data: {}\n{}".format(x_data.shape, x_data))
print("y_data: {}\n{}".format(y_data.shape, y_data))

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])
W = tf.Variable(tf.zeros([3, 3]))

h = tf.matmul(X, W)
H = tf.nn.softmax(h)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(H), 1))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

session = tf.Session()
session.run(init)

for step in range(2001):
	session.run(train, feed_dict={X: x_data, Y: y_data})
	if step % 200 == 0:
		print("{:4} {:8.6}\n{}".format(step, session.run(cost, feed_dict={X: x_data, Y: y_data}), session.run(W)))

print("---------------------------------------------")

testSet = [[[1, 11, 7]],[[1, 3, 4]],[[1, 1, 0]]]

for t in testSet:
	result = session.run(H, feed_dict={X: t})
	print("{}: {}".format(t, session.run(tf.argmax(result, 1))))

result = session.run(H, feed_dict={X: [testSet[0][0], testSet[1][0], testSet[2][0]]})
print(session.run(tf.argmax(result, 1)))

session.close()
