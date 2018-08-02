#!/usr/bin/python

from __future__ import print_function
import tensorflow as tf
import numpy as np

data = np.loadtxt("dataset/trainDatasetForXOR", unpack=True)

x_data = np.transpose(data[:-1])	# 4 * 2 
y_data = np.reshape(data[-1], (4, 1))

print(x_data.shape, x_data)
print(y_data.shape, y_data)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 2], -1., 1.))
W2 = tf.Variable(tf.random_uniform([2, 1], -1., 1.))

B1 = tf.Variable(tf.zeros([2]))
B2 = tf.Variable(tf.zeros([1]))

L2 = tf.sigmoid(tf.matmul(X, W1) + B1)
H = tf.sigmoid(tf.matmul(L2, W2) + B2)

cost = -tf.reduce_mean(Y * tf.log(H) + (1 - Y) * tf.log(1 - H))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

s = tf.Session()
s.run(init)

for step in range(10001):
	s.run(train, feed_dict={X: x_data, Y: y_data})
	if step % 1000 == 0:
		r1, (r2, r3) = s.run(cost, feed_dict={X: x_data, Y: y_data}), s.run([W1, W2])
		print("{:5} {:10.8f} {} {}".format(step, r1, np.reshape(r2, (1, 4)), np.reshape(r3, (1, 2))))

print("-"*50)

correct_prediction = tf.equal(tf.floor(H + 0.5), Y)

# Calculate Accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
param = [H, tf.floor(H + 0.5), correct_prediction, accuracy]
result = s.run(param, feed_dict={X: x_data, Y: y_data})

print(result[0])
print(result[1])
print(result[2])
print(result[-1])
print("Accuracy:", accuracy.eval({X: x_data, Y: y_data}, session=s))
