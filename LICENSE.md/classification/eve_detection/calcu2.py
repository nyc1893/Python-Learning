

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
            if(np.argmax(df2[j])>30*i and np.argmax(df2[j])<=30*(i+1)):
                a[i] +=1
    res= -1
    max =1 
    # print("a",a)
    for i in range(180):
        if(a[i]>max):
            res = i
            max = a[i]
    if(res==-1):
        ss = np.max(df2,axis=1)
        id = np.argmax(df2[np.argmax(ss)])
        print("id",id)
        for i in range(180):
            if(id>30*i and id<=30*(i+1)):
                return i
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
def remove_z(str):
    if str[0] == '0':
        return str[1]
    else:
        return str
        
def rm3(X,y):
    """
    THIS FUNCTION REMOVES THE PLANNED EVENTS FROM THE EVENT DATASET
    """
    
    X_new=[]
    y_new=[]
    word =[]
    df = pd.read_csv("../../../../../muti.csv")
    df = df["new"].values.tolist()
    # print(len(df))
    # print(df[:5])
    for i in range(len(y)):
        #print(i)
        temp = y[i,2].split("_")
        ww = temp[0]+"_"+temp[1]+"_"+remove_z(temp[2])+"_"+temp[3]
        # print(np.isin(ww,df))

        if(np.isin(ww,df)==False):
        
            if y[i,0]==0:
                y_new.append(0)
                X_new.append(X[i,:,:,:])
                # word.append(y[i,0])
                
                
            if y[i,0]==1:
                y_new.append(1)
                X_new.append(X[i,:,:,:])
                # word.append(y[i,0])
        
                
            elif y[i,0]==2:
                y_new.append(2)
                X_new.append(X[i,:,:,:])
                # word.append(y[i,0])
            
            elif y[i,0]==3:
                y_new.append(3)
                X_new.append(X[i,:,:,:])
                # word.append(y[i,0])
            elif y[i,0]==4:
                y_new.append(0)
                X_new.append(X[i,:,:,:])
                # word.append(y[i,0])
            elif y[i,0]==5:
                y_new.append(1)
                X_new.append(X[i,:,:,:])
                # word.append(y[i,0])

    return  np.array(X_new), np.array(y_new)

def proces(data,k):
    st1 =[]
    data = np.squeeze(data)
    
    for j in range(0,data.shape[0]):
        st2=[]
        for i in range(0,data.shape[1]):
            if(k == 0):
                temp = data[j,i,:]/np.mean(data[j,i,:])
            elif(k == 1):
                temp = np.deg2rad(data[j,i,:])
            elif(k == 2):
                temp = data[j,i,:]/100             
            elif(k == 3):
                temp = np.deg2rad(data[j,i,:])                
            elif(k == 4):
                temp = data[j,i,:]/np.mean(data[j,i,:])            
            elif(k == 5):
                temp = data[j,i,:]      
            st2.append(temp)
        st1.append(st2)

    st1 = np.array(st1)
    st1 = st1[:,:,np.newaxis,:]
    # print(st1.shape)
    return st1
    
def rd3(ii,k):

    path1 = '../../../pickleset1/'
    list = ['vp_m','vp_a','ip_m','ip_a','f','rocof']
    if(ii!=12):
        p1 = open(path1 +'X_S'+str(ii)+'_'+str(list[k])+'_6.pickle',"rb")
        pk1 = pickle.load(p1)
        
        pk3  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[k])+'_6.csv')
   
    else:
        p1 = open(path1 +'X_S'+str(ii)+'_'+str(list[k])+'_61.pickle',"rb")
        pk1 = pickle.load(p1)
        
        p1 = open(path1 +'X_S'+str(ii)+'_'+str(list[k])+'_62.pickle',"rb")
        pk2 = pickle.load(p1)   
        
        
        pk3  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[k])+'_61.csv')
        pk4  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[k])+'_62.csv')
        
        pk1= np.concatenate((pk1, pk2), axis=0)
        pk3= pd.concat([pk3, pk4], axis=0)
        
    # X_train = pk1[:,:,1*60:91*60,]
    X_train = pk1[:,:,(300-29)*60:(300+61)*60,]
    
    y_train = pk3.values
    
    X_train, y_train  = rm3(X_train, y_train)
    X_train=X_train.transpose(0,1,3,2)
    # print(X_train.shape)
    
    X_train = proces(X_train,k) 
    
    # print(X_train.shape)
    return X_train, y_train 
    
    
def plot_eve2(ii):
    path1 ="../event/"
    p1 = open(path1 +'90s_X_S'+str(ii)+'.pickle',"rb")
    df1 = pickle.load(p1)    
    print(df1.shape)
    pk3  = pd.read_csv(path1 +'y_S'+str(ii)+'.csv').values
    a,_ =rd3(ii,0)
    for i in range(1,6):
        b,_ =rd3(ii,i)
        a= np.concatenate((a, b), axis=2)
        
    ca=["Line","Trans","Freq","Oscillation"]
    res =[]
    v = 0
    df = pd.read_csv("../run_bo/err.csv")
    df = df[df["real"]==1 ]
    dt = df[df["s"]==ii]
    cc = dt["NO"].values.tolist()    
    
    # for j in range(df1.shape[0]):
    for j in cc:
    # for j in range(136,136+1):
        ll =[]

        # if(pk3[j]==label and v<20):
        temp = df1[j]

        mod = get_ind(temp)
        print("mod",mod)
        ind = range(30*(mod),30*(mod+1))
        # print(ind)
        plt.figure( figsize=(15,12))
        for i in range(0,8):
            # ind2 = np.argmax(df1[j,i,:])
            plt.subplot(7,2,i+1)
            plt.plot(range(temp.shape[1]),temp[i])
            # plt.scatter(ind2,temp[i,ind2],marker='p',c='',edgecolors='r',zorder=10)
            plt.scatter(ind,temp[i,ind],marker='p',c='',edgecolors='r',zorder=10)
            # plt.xlabel(str(round(np.max(temp[i,ind])/np.mean(temp[i]),3)))
            # plt.xlabel(word[i])
            plt.xlabel(word[i]+",max index:"+str(np.argmax(temp[i])))
        for i in range(6):
            plt.subplot(7,2,i+9)
            for l in range(23):
                plt.plot(range(a[j,l,i,:].shape[0]),a[j,l,i,:])
                plt.scatter(ind,a[j,l,i,ind],marker='p',c='',edgecolors='r',zorder=10)
            plt.xlabel(word[i])
            
        plt.title("event type: "+ca[int(pk3[j])])
        plt.tight_layout()             
        plt.savefig("../plot/S"+str(ii)+"_eve_"+ca[int(pk3[j])]+"_No_"+str(j))  
        v+=1
        
    
    
def plot_nor2(ii):
    path1 ="../nor/"
    p1 = open(path1 +'90s_X_S'+str(ii)+'.pickle',"rb")
    df1 = pickle.load(p1)    
    
    pk3  = pd.read_csv(path1 +'y_S'+str(ii)+'.csv').values
    print(df1.shape)
    print(pk3.shape)
    
    a,_ =rd3(ii,0)
    for i in range(1,6):
        b,_ =rd3(ii,i)
        a= np.concatenate((a, b), axis=2)
    # print(a.shape)
    # print(a[j,l,i,:].shape[0])
    # print(a[j,l,i,:].shape)
    df = pd.read_csv("../run_bo/err.csv")
    df = df[df["real"]==0 ]
    dt = df[df["s"]==ii]
    cc = dt["NO"].values.tolist()
    res =[]
    # df1.shape[0]
    # for j in range(cc):
    for j in cc:
        print(j)
        ll =[]
        y = pk3[j]
        temp = df1[j]

        mod = get_ind2(temp)
        # print(mod)
        ind = range(30*(mod),30*(mod+1))
        if(mod!=-1):
            plt.figure( figsize=(15,12))
            for i in range(0,8):
                plt.subplot(7,2,i+1)
                plt.plot(range(temp.shape[1]),temp[i])
                plt.scatter(ind,temp[i,ind],marker='p',c='',edgecolors='g',zorder=10)
                # plt.xlabel(str(round(np.max(temp[i,ind])/np.mean(temp[i]),3)))
                plt.xlabel(word[i]+",max index:"+str(np.argmax(temp[i])))
            
            for i in range(6):
                plt.subplot(7,2,i+9)
                for l in range(23):
                    plt.plot(range(a[j,l,i,:].shape[0]),a[j,l,i,:])
                    plt.scatter(ind,a[j,l,i,ind],marker='p',c='',edgecolors='g',zorder=10)
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
    # plot_eve2(1,3)
    # plot_eve(1)
    plot_eve2(1)
    plot_eve2(2)
    # top()
    # fun(1)
    s2 = timeit.default_timer()
    print('Time: ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

