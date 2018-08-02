#!/usr/bin/python

import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
output = tf.mul(a, b)

with tf.Session() as s:
	result = s.run([output], feed_dict={a:[7.], b:[2.0]})
	print(result)
