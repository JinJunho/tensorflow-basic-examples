#!/usr/bin/python

import tensorflow as tf

state = tf.Variable(0)
init_op = tf.initialize_all_variables()

with tf.Session() as s:
	s.run(init_op)
	print(s.run(state))
