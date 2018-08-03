#!/usr/bin/python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# http://pythonkim.tistory.com/56

batch_size = 128
test_size = 256
stride = [1, 1, 1, 1]
pool_stride = [1, 2, 2, 1]
ksize = [1, 2, 2, 1]
padding = "SAME"

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, W1, W2, W3, W4, WO, p_keep_conv, p_keep_hidden):
	# X: ? 28 28 1
	# W1: 3 3 1 32
	l1a = tf.nn.relu(tf.nn.conv2d(X, W1, strides=stride, padding=padding))	# ? 28 28 32
	l1m = tf.nn.max_pool(l1a, ksize=ksize, strides=pool_stride, padding=padding)	# ? 14 14 32
	l1 = tf.nn.dropout(l1m, p_keep_conv)

	# W2: 3 3 32 64
	l2a = tf.nn.relu(tf.nn.conv2d(l1, W2, strides=stride, padding=padding))
	l2m = tf.nn.max_pool(l2a, ksize=ksize, strides=pool_stride, padding=padding)	# ? 7 7 64
	l2 = tf.nn.dropout(l2m, p_keep_conv)

	# W3: 3 3 64 128
	l3a = tf.nn.relu(tf.nn.conv2d(l2, W3, strides=stride, padding=padding))
	# Max pooling adds zero paddings when shrinked dimension is not divided by stride size
	l3m = tf.nn.max_pool(l3a, ksize=ksize, strides=pool_stride, padding=padding)	# ? 4 4 128
	l3r = tf.reshape(l3m, [-1, W4.get_shape().as_list()[0]])	# (?, 2048) Fully Connected
	l3 = tf.nn.dropout(l3r, p_keep_conv)

	# W4: 2048, 625
	l4a = tf.nn.relu(tf.matmul(l3, W4))	# ? 625
	l4 = tf.nn.dropout(l4a, p_keep_hidden)

	# WO: 625 10
	pyx = tf.matmul(l4, WO)	# ? 10
	return pyx

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
	# you need to initialize all variables
	tf.initialize_all_variables().run()

	for i in range(100):
		training_batch = zip(range(0, len(trX), batch_size),
												range(batch_size, len(trX)+1, batch_size))
		for start, end in training_batch:
			sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
							p_keep_conv: 0.8, p_keep_hidden: 0.5})

	test_indices = np.arange(len(teX)) # Get A Test Batch
	np.random.shuffle(test_indices)
	test_indices = test_indices[0:test_size]

	print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
		sess.run(predict_op, feed_dict={X: teX[test_indices],
			Y: teY[test_indices],
			p_keep_conv: 1.0,
			p_keep_hidden: 1.0})))
