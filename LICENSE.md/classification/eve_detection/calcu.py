

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

def fun(ii):

    df1 = np.load('event_S_'+str(ii)+'.npy')
    print(df1.shape)
    st =[]
    for j in range(df1.shape[0]):
        kk = np.zeros(24)
        for i in range(0,8):
            temp = df1[j,i,:]
            kk[i*3+0] =np.max(temp)
            kk[i*3+1] =np.mean(temp)
            kk[i*3+2] =np.std(temp)
        st.append(kk)
    st= np.array(st)
    df =pd.DataFrame(st)
    list1 = ["max","avg","std"]
    list2 = [*range(1,9)]
    list3 = []

    for j in range(len(list2)):
        for i in range(len(list1)):
            list3.append(str(list2[j])+list1[i])
    df.columns = list3
    # df["label"] = y.values
    # print(df.head())
    print(df.shape)
    df.to_csv("Event_S"+str(ii)+".csv",index = None)
    print("save done No." +str(ii))



def fun2(ii):

    df1 = np.load('non_S_'+str(ii)+'.npy')
    print(df1.shape)
    st =[]
    for j in range(df1.shape[0]):
        kk = np.zeros(24)
        for i in range(0,8):
            temp = df1[j,i,:]
            kk[i*3+0] =np.max(temp)
            kk[i*3+1] =np.mean(temp)
            kk[i*3+2] =np.std(temp)
        st.append(kk)
    st= np.array(st)
    df =pd.DataFrame(st)
    list1 = ["max","avg","std"]
    list2 = [*range(1,9)]
    list3 = []

    for j in range(len(list2)):
        for i in range(len(list1)):
            list3.append(str(list2[j])+list1[i])
    df.columns = list3
    # df["label"] = y.values
    # print(df.head())
    print(df.shape)
    df.to_csv("NonEvent_S"+str(ii)+".csv",index = None)
    print("save done No." +str(ii))

    
def main():
    s1 = timeit.default_timer()  
    for ii in range(1,14):
        fun(ii)
        fun2(ii)
    s2 = timeit.default_timer()
    print('Time: ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

