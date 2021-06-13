

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
    path1 ="../event/"
    p1 = open(path1 +'90s_X_S'+str(ii)+'.pickle',"rb")
    df1 = pickle.load(p1)    
    print(df1.shape)

    res =[]
    for j in range(df1.shape[0]):
    
        temp = df1[j]

        mod = get_ind(temp)
        # print(mod)
        ind = range(30*(mod),30*(mod+1))
        ll =[]
        for i in range(0,8):
            ll.append(math.ceil(np.argmax(df1[j,i,:])/30))
        # zuida.append(np.argmax(df1[0,i,:]))

        res.append(df1[j,:,ind])
    res = np.array(res)
    
    np.save("event_S_"+str(ii)+".npy",res)
        
def fun2(ii):
    path1 ="../nor/"
    p1 = open(path1 +'90s_X_S'+str(ii)+'.pickle',"rb")
    df1 = pickle.load(p1)    
    print(df1.shape)

    # df1.shape[0]
    res =[]
    for j in range(df1.shape[0]):
        temp = df1[j]

        mod = get_ind2(temp)
        # print(mod)
        ind = range(30*(mod),30*(mod+1))
        
        res.append(df1[j,:,ind])
    res = np.array(res)
    
    np.save("non_S_"+str(ii)+".npy",res)
    
def get_ind(temp):
    select =[0,2,4,5,6,7]
    df2 = temp[select,:]
    a = np.zeros(180)
    for j in range(6):
        # print("np.argmax(df2[j])",np.argmax(df2[j]))
        for i in range(180):
            if(np.argmax(df2[j])>30*i and np.argmax(df2[j])<30*(i+1)):
                a[i] +=1
    res= 0
    max =0 
    # print("a",a)
    for i in range(180):
        if(a[i]>max):
            res = i
            max = a[i]
    return int(res)
    
def get_ind2(data):
    select =[0,2,4,5,6,7]
    df2 = data[select,:]


    df3 = np.mean(df2, axis=1)
    # print(df3[0])
    # print("df3.shape",df3.shape)
    # print("df2.shape",df2.shape)
    i = 1
    for i in range(180):
        temp = df2[:,30*i:30*(i+1)]
        # print("temp.shape",temp.shape)
        df4 = np.max(temp, axis=1)
        # print(df4)
        df4 = 5*df3-df4
        # print(i)
        # print(df4)
        if((df4>0).all()):
            
            return i
        # print(df4)
 

    return -1
        
    
    
    
def plot_eve(ii):
    path1 ="../event/"
    p1 = open(path1 +'90s_X_S'+str(ii)+'.pickle',"rb")
    df1 = pickle.load(p1)    
    print(df1.shape)
    pk3  = pd.read_csv(path1 +'y_S'+str(ii)+'.csv').values
    # df1.shape[0]
    ca=["Line","Trans","Freq","Oscillation"]
    res =[]
    v = 0
    for j in range(df1.shape[0]):
        ll =[]

        if(pk3[j]==0 and v<10):
            temp = df1[j]

            mod = get_ind(temp)
            # print(mod)
            ind = range(30*(mod),30*(mod+1))
            # print(ind)
            plt.figure( figsize=(15,10))
            for i in range(0,8):
                ind2 = np.argmax(df1[j,i,:])
                plt.subplot(4,2,i+1)
                plt.plot(range(temp.shape[1]),temp[i])
                # plt.scatter(ind2,temp[i,ind2],marker='p',c='',edgecolors='r',zorder=10)
                plt.scatter(ind,temp[i,ind],marker='p',c='',edgecolors='r',zorder=10)
                # plt.xlabel(str(round(np.max(temp[i,ind])/np.mean(temp[i]),3)))
                plt.xlabel(word[i])
                
            plt.title("event type: "+ca[int(pk3[j])])
            plt.tight_layout()             
            plt.savefig("../plot/S"+str(ii)+"_eve_"+ca[int(pk3[j])]+"_No_"+str(j))  
            v+=1
    
word =["vpm","vpa","ipm","ipa","freq","Rocof","Active","Reactive"]

def plot_nor(ii):
    path1 ="../nor/"
    p1 = open(path1 +'90s_X_S'+str(ii)+'.pickle',"rb")
    df1 = pickle.load(p1)    
    
    pk3  = pd.read_csv(path1 +'y_S'+str(ii)+'.csv')
    print(df1.shape)

    # df1.shape[0]
    res =[]
    # df1.shape[0]
    for j in range(20):
        print(j)
        ll =[]
        y = pk3[j]
        temp = df1[j]

        mod = get_ind2(temp)
        # print(mod)
        ind = range(30*(mod),30*(mod+1))
        if(mod!=-1):
            plt.figure( figsize=(15,10))
            for i in range(0,8):
                plt.subplot(4,2,i+1)
                plt.plot(range(temp.shape[1]),temp[i])
                plt.scatter(ind,temp[i,ind],marker='p',c='',edgecolors='g',zorder=10)
                # plt.xlabel(str(round(np.max(temp[i,ind])/np.mean(temp[i]),3)))
                plt.xlabel(word[i])
            plt.tight_layout()             
            plt.savefig("../plot/S"+str(ii)+"_Non-Event_"+str(j))  
            print("plot out "+str(j))
                
def top():
    for i in range(1,14):
        fun(i)
        fun2(i)
        
def main():
    s1 = timeit.default_timer()  
    plot_eve(1)
    # plot_nor(1)
    # top()
    # fun(1)
    s2 = timeit.default_timer()
    print('Time: ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

