import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

n_steps = 3
n_inputs = 3
n_neurons = 5


X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
seq_length = tf.placeholder(tf.int32, [None])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, sequence_length=seq_length, dtype=tf.float32)
# X_batch = np.array([
  # t = 0      t = 1
  # [[0, 1, 2], [9, 8, 7]], # instance 0
  # [[3, 4, 5], [0, 0, 0]], # instance 1
  # [[6, 7, 8], [6, 5, 4]], # instance 2
  # [[9, 0, 1], [3, 2, 1]], # instance 3
# ])
X_batch = np.array([
					[[1,2,3],[4,5,6],[7,8,9]]
					])
# X_batch = X_batch.reshape((-1, 3, 3))
print(X_batch.shape)
seq_length_batch = np.array([2])

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outputs_val, states_val = sess.run([outputs, states], 
                                     feed_dict={X: X_batch, seq_length: seq_length_batch})

  print(outputs_val)
  print()
  print(states_val)
