

# import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import math
import pandas as pd
import time  
from sklearn import metrics  
import pickle as pickle  
import timeit
import matplotlib.pyplot as plt

def get_ind(temp):
    select =[0,2,4,5,6,7]
    df2 = temp[select,:]
    b = np.zeros(6)
    a =[]
    for j in range(6):
        a.append(np.argmax(df2[j]))
    a.sort()
    
    for i in range(0,6):
        ind = a[i]
        b[i] = num_contain(ind,a)
        
    max = 1
    c = -1
    res =-1
    for i in range(0,6):
        if(b[i]>max):
            max= b[i]
            c=i
            res = a[c]  
           
    if(res==-1):
        ss = np.max(df2,axis=1)
        id = np.argmax(df2[np.argmax(ss)])
        res = id 
    if(res>5340-29):
        res = 5340-29
    print(res)
    return (res-1)
    
def num_contain(ind,a):
    win=30
    low = ind-1
    high = low+win
    ll = []
    sum =0
    for i in range(6):
        if(a[i]>low and a[i]<high):
            sum+=1
            
    return sum
    
def save_file(ii):
    path1 ="../dif_scal/60/"
    p1 = open(path1 +'X_S'+str(ii)+'.pickle',"rb")
    pk1 = pickle.load(p1)
    print(pk1.shape)
    p2 = pd.read_csv(path1 +"y_S"+str(ii)+".csv")

    # print(p2.shape)
    st = []
    for j in range(pk1.shape[0]):
        temp = pk1[j]
        ind2 = get_ind(temp)
        ind = range(ind2,ind2+30)        
        # print(ind2)
        st.append(temp[:,ind])
    st = np.array(st)
    # print(st.shape)

    np.save('event_S_'+str(ii)+'.npy',st)
    print("S"+str(ii)+" save done")

def top():
    for i in range(1,14):
        save_file(i)
def main():
    s1 = timeit.default_timer()  

    top()
    s2 = timeit.default_timer()
    print('Time: ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

