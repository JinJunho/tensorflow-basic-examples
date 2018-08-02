#!/usr/bin/python

import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

session = tf.Session()
session.run(init)

tf.reset_default_graph()

for step in range(1000):
	session.run(train)
	if step % 100 == 0:
		print(step, session.run(W), session.run(b))

writer = tf.train.SummaryWriter("/tmp/tensor", session.graph)
session.close()
