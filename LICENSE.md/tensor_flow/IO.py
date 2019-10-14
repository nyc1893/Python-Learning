import tensorflow as tf

x = tf.constant([1,2,3])

with tf.Session() as sess:
	print(sess.run(x))
