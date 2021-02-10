

import numpy as np

import pandas as pd
import sys
import heapq
from pyts.image import MarkovTransitionField

def gen():
    temp=np.random.rand(100,23,200,2)
    print(temp.shape)
    return temp   
    
def select():
    X=gen()
    res =[]
    for j in range(0,X.shape[0]):
        nums = []
        for i in range(0,23):
            temp = X[j,i,:,0]
            nums.append(np.max(temp)-np.min(temp))
            
        max_num_index_list = map(nums.index, heapq.nlargest(3, nums))
        ll = list(max_num_index_list)
        temp = X[j,ll,:,:]
        res.append(temp)
    return  np.array(res)

def MTF():
    temp =select()
    print(temp.shape)
    t1 =[]
    t2 =[]
    mtf = MarkovTransitionField(image_size=120,strategy='quantile',n_bins=10)
    for k in range(0,5):
        st1 =[]
        st2 =[]
        for j in range(0,temp.shape[1]):
 
            x = temp[k,j,:,0]
            temp1 = mtf.fit_transform(temp.reshape(-1,x.shape[0]))
            x = temp[k,j,:,0]
            temp2 = mtf.fit_transform(temp.reshape(-1,x.shape[0]))

            st1.append(temp1[0])
            st2.append(temp2[0])
        t1.append(st1)
        t2.append(st2)
    st1 = np.array(t1)
    st2 = np.array(t2)
        
    print(st1.shape)
    print(st2.shape)
    
    st1 = st1.reshape(-1,120,120,st1.shape[1])
    st2 = st2.reshape(-1,120,120,st2.shape[1])
    print(st1.shape)
    print(st2.shape)
        
def main():
    MTF()

if __name__ == '__main__':  

    main()
