#!/usr/bin/python

import tensorflow as tf
import matplotlib.pyplot as plt

x = [1., 2., 3.]
y = [1., 2., 3.]
m = len(x)

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

hypothesis = W * x + b

cost = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

session = tf.Session()
session.run(init)


for step in range(1000):
	session.run(train)
	if step % 100 == 0:
		print("{}: {} {}".format(step, session.run(W), session.run(b)))

session.close()
