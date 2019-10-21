"""
This is for EVT kdd 2019 paper

"""
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn

data1=pd.read_csv('data\\dge_2009.csv')
data2=pd.read_csv('data\\ge_2009.csv')

m = 6 # No. of memory size
def getdata(data1,data2,t,m):
	


	v = data1.loc[0:t].values
	x = data2.loc[0:t].values

	

	len2 = []
	ind1 = []
	ind2 = []
	v2 =[]
	pr0 =[]
	pr1 =[]
	Vwm=0
	i = 0
	while i< m:
		c1 = (random.randint(0,x.shape[0]))
		c2 =  (random.randint(0,x.shape[0]))
		
		if c2>c1:
			# print(c1,c2)
			Wm = x[c1:c2]
			if (np.where(v[c1:c2] != 0)[0]).size == 0:
				Vwm = 0
			else:
				Vwm = 1
			# print("\nwindow value:\n",Wm[:])
			# print("\nwindow index1:",c1)
			# print("\nwindow index2:",c2)
			# print("Extreme indicator",Vwm)
			p0 =  v[c1:c2][v[c1:c2]==0].size 
			p1 =  v[c1:c2][v[c1:c2]!=0].size 
			# print("p0=",(p0/(p0+p1)))
			# print("p1=",(p1/(p0+p1)))
			i=i+1
			len2.append(Wm.shape[0])
			# w2.append(Wm[:])
			ind1.append(c1)
			ind2.append(c2)
			v2.append(Vwm)
			pr0.append(p0/(p0+p1))
			pr1.append(p1/(p0+p1))
		
	return np.array(len2),np.array(ind1),np.array(ind2),np.array(v2),np.array(pr0),np.array(pr1)
	
def gru_whole(X_batch,n_steps,n_inputs,n_neurons):
	# for the whole windows
	tf.reset_default_graph() 
	# print(n_inputs)
	X_batch = X_batch.reshape((-1, n_steps, n_inputs))
	X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
	seq_length = tf.placeholder(tf.int32, [None])
	basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
	o1, s1 = tf.nn.dynamic_rnn(basic_cell, X, sequence_length=seq_length, dtype=tf.float32)	
	seq_length_batch = np.array([1])

	with tf.Session() as sess:
	  sess.run(tf.global_variables_initializer())
	  outputs_val, states_val = sess.run([o1, s1], 
										 feed_dict={X: X_batch, seq_length: seq_length_batch})

	  # print("outputs_val",outputs_val)
	  # print()
	  return states_val	
	
def gru_win(X_batch,n_steps,n_inputs,n_neurons,m):
	tf.reset_default_graph() 
# for window sj
	mem_s = 0
	a4 = np.zeros(shape=(1,n_inputs))
	
	for k in range(0,m):
		if (len2[k]<n_inputs):
			# print('xiaoyu')
			x2 = x[ind1[k]:ind2[k]]
			for _ in range(0,n_inputs-len2[k]):
				x2 = np.append(x2,0)

		else:
			x2 = x[ind1[k]:ind2[k]]
		a4 = np.append(a4,x2)
	a4 = a4.reshape(-1,max(len2)) 
	a4 = np.delete(a4,0,axis =0) 
	X_batch = a4 
	X_batch = X_batch.reshape((-1, n_steps, n_inputs))
	print('X_batch',X_batch)
	print('X_batch shape',X_batch.shape)
	X2 = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
	seq_length2 = tf.placeholder(tf.int32, [None])
	basic_cell2 = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
	o2, s2 = tf.nn.dynamic_rnn(basic_cell2, X2, sequence_length=seq_length2, dtype=tf.float32)
	

	seq_length_batch2 = np.ones(m)


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		outputs_val, states_val = sess.run([o2, s2], 
										 feed_dict={X2: X_batch, seq_length2: seq_length_batch2})

		# print("outputs_val",outputs_val)
		# print()
		# print("states_val",states_val)
		mem_s = np.append(mem_s,states_val)
	mem_s = np.delete(mem_s, 0)	
	mem_s = mem_s.reshape((-1, n_neurons))
	return mem_s
	
	
t = 10 # time slot
len2,ind1,ind2,v2,pr0,pr1 = getdata(data1,data2,t,m)
v = data1.loc[0:t].values
x = data2.loc[0:t].values

k = 0


print('max(len2)',max(len2))

X_batch = x;

n_steps = 1
n_inputs = t+1
n_neurons = 5  # feature numbers


	  
print('whole:',gru_whole(X_batch,n_steps,n_inputs,n_neurons))
print()

	
n_steps = 1
n_inputs = max(len2)
print(n_inputs)
n_neurons = 5  # feature numbers
out =  gru_win(X_batch,n_steps,n_inputs,n_neurons,m)


print('gru_win', out)
print("outshape",out.shape)

out = out.reshape(-1,1,n_neurons)



h = tf.transpose(out, [1, 0, 2])
print("h.shape:",tf.shape(h))
# Define weights
weights = {
    'out': tf.Variable(tf.truncated_normal([n_neurons, 1], stddev=1.0))
}
biases = {
    'out': tf.Variable(tf.truncated_normal([1], stddev=0.1))
}


# tf Graph input
lr = tf.placeholder(tf.float32, [])
X = tf.placeholder(tf.float32, [None, n_steps, n_neurons])
# x2 = tf.placeholder(tf.float32, [None, n_steps, n_input])
Y = tf.placeholder(tf.float32, [None, 1])


pred = tf.nn.bias_add(tf.matmul(tf.cast(h[-1], tf.float32) , tf.cast(weights['out'], tf.float32)), biases['out'])


# Define loss (Euclidean distance) and optimizer

gama = tf.constant(1.0,dtype=tf.float32)
one  = tf.constant(1.0,dtype=tf.float32)

two = tf.placeholder(tf.float32, [None, 1])

EVL =  tf.multiply(two,tf.pow(tf.subtract(one,tf.divide(pred, gama)),gama))

# print("pr0 shape",pr0.shape)
pr0 = pr0.reshape((-1,1))
# print("pr0 shape",pr0.shape)
# print("pr0 ",pr0)
individual_losses = tf.reduce_sum(EVL, reduction_indices=1)
loss = tf.reduce_mean(individual_losses)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()
n_outputs = 1

# Parameters
learning_rate = 0.005
training_iters = 100
batch_size = 1
training_iter_step_down_every = 250000
display_step = 20

	
with tf.Session() as sess:
    sess.run(init)
    step = 1

    loss_value = float('+Inf')
    target_loss = 1e-10
    current_learning_rate = learning_rate    

    # Keep training until reach max iterations
    while (step * batch_size < training_iters) and (loss_value > target_loss):
        current_learning_rate = learning_rate
        current_learning_rate *= 0.1 ** ((step * batch_size) // training_iter_step_down_every)

        # _, batch_x, __, batch_y = generate_sample(f=None, t0=None, batch_size=batch_size, samples=n_steps,
                                                  # predict=n_outputs)
        batch_x = out
        batch_y = v2
        batch_two = pr0
        # print("batch_x:",batch_x.size)
         
        # batch_x = batch_x.reshape((-1, n_steps, n_input))
        batch_y = batch_y.reshape((-1, n_outputs))
        # batch_two = batch_two.reshape((-1, n_outputs))
        # print("batch_two:",batch_two)

        fd1 = {X: batch_x, Y: batch_y,two:batch_two,lr: current_learning_rate}
        sess.run(optimizer, feed_dict = fd1)
        if step % display_step == 0:
            # Calculate batch loss
            fd = {X: batch_x, Y: batch_y,two:batch_two}
            loss_value = sess.run(loss, feed_dict=fd)
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss_value))
            # print(pred.shape.as_list())
            # print(pred)

        step += 1
    print("Optimization Finished!")
    # s2 = timeit.default_timer()  
    # print ('Runing time is Hour:',round((s2 -s1)/3600,2))
	
	
	
	
	



"""
# V 解决重复定义问题  tf 
# 利用tf.reset_default_graph()重置

# V解决第一part traning section

# 解决第2 part traning section


# V loss  中 加入 feed
"""

