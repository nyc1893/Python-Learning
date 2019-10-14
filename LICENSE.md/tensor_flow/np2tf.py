import tensorflow as tf
import numpy as np

a=np.array([[1,2,3],[4,5,6],[7,8,9]])

print (a)
b=tf.constant(a)
tensor_a=tf.convert_to_tensor(a)
with tf.Session() as sess:
    print (b)
    for x in b.eval():      #b.eval()就得到tensor的数组形式
        print (x)
		
    print (tensor_a)
    for x in tensor_a.eval():      #b.eval()就得到tensor的数组形式
        print (x)
