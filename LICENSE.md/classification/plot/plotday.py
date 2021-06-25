

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
from scipy.linalg import svd 
# import time
import timeit
from random import randint
# import math


def rd3(ii,k):

    path1 = '../labeled/'
    list = ['vp_m','vp_a','ip_m','ip_a','f','rocof']

    p1 = open(path1 +'X_S'+str(ii)+'_'+str(list[k])+'_6.pickle',"rb")
    pk1 = pickle.load(p1)
        
    X_train = pk1   

    X_train=X_train.transpose(0,1,3,2)

    # X_train = proces(X_train,k) 
    
    return X_train


def lay1(ii):  
    k =0
    a = rd3(ii,k)
    for k in range(1,5+1):
        a2 = rd3(ii,k)
        a = np.concatenate((a, a2), axis=2)  
    st = [] 
    print(a.shape)
    return a 
    
def lay2(ii):  

    b = lay1(ii)

    a = b[0,:,:,:]
    print(a.shape)
    for j in range(1,a.shape[0]):
        a2 = b[j,:,:,:]
        a = np.concatenate((a, a2), axis=2)  

    print(a.shape)
    return a 
    
    
def proces(ii):

    st1 =[]
    data = lay2(ii)
    
    for j in range(0,data.shape[0]):
        st2=[]
        for i in range(0,data.shape[1]):
            if(i == 0):
                temp = data[j,i,:]/np.mean(data[j,i,:])
            elif(i == 1):
                temp = np.deg2rad(data[j,i,:])
            elif(i == 2):
                temp = data[j,i,:]/100             
            elif(i == 3):
                temp = np.deg2rad(data[j,i,:])                
            elif(i == 4):
                temp = data[j,i,:]/np.mean(data[j,i,:])            
            elif(i == 5):
                temp = data[j,i,:]      
            st2.append(temp)
        st1.append(st2)

    st1 = np.array(st1)
    # st1 = st1[:,:,np.newaxis,:]
    print(st1.shape)
    return st1

word =["vpm","vpa","ipm","ipa","freq","Rocof","Active","Reactive"]

def rd_raw(ii,k):
    word2 =["vp_m","vp_a","ip_m","ip_a","f","rocof"]
    path1 ="../lab2/"
    p1 = open(path1 +'X_S'+str(ii)+'_'+str(word2[k])+'_6.pickle',"rb")
    df1 = pickle.load(p1)        
    # print(df1.shape)
    st=[]
    for j in range(df1.shape[1]):
        # st2=[]
        a = df1[0,j,:,0]
        # df1.shape[0]
        for i in range(1,df1.shape[0]):
            a2 = df1[i,j,:,0]
            a = np.append(a,a2)
            
        st.append(a)
    st = np.array(st)
    
    print(st.shape)
    return st
    


# def raw(k):

    # a = rd_raw(1,0)
    # for k in range(2,5):
        # a2 = rd_raw(1,k)
        # a =  np.append(a,a2,axis = 1)
    # print(a.shape)
    # return a 

def get_single(x,t_window):



    # plt.plot(xv_a[0,:])
    # plt.show()
    # plt.plot(xi_a[0,:])
    # plt.show()

    # t_window = 30
    miss = 0.2
    # remove nan values
    inds=np.where(np.isnan(x))
    col_mean=[]
    x[inds]=0
    # inserting nan value with interopolation of 5 closest values
    for ii in range(len(inds[0])):
       col_mean.append(np.mean(x[inds[0][ii],inds[1][ii],inds[2][ii]-5:inds[2][ii]+5]))
       x[inds[0][ii],inds[1][ii],inds[2][ii]]=col_mean[ii]
       
    np.random.seed(0)   
                        
    lstnan = np.isnan(x)
    x = np.nan_to_num(x)




      
    error = 0.01
    errList_OLAP = []
    time_OLAP = []
    Omg_cal = []

    Zeta_m =[]
    
    for m in range(0,1):
        eta = []
        zeta = []
        X=x
        X_mat = X.T
        for i in range(X_mat.shape[0]):
    #        if i % 100 ==0:
    #            print('Iteration '+ str(i)+ ' for measurement ' +str(m))
    #         Intialze the first matrix with no missing entries
            if i == 0:
                Rec_M = X_mat[0:t_window,:]
                Orig_M = Rec_M
                Rec_OLAP = Rec_M
                M_w = X_mat[0:t_window,:]
                for i_k in range(t_window):
                    eta.append(0)
                
            t = i + t_window # proceed with the new sample index
            
            if t == X_mat.shape[0]:
                break
            
            # svd calculations for the old matrix
            u, s, vh = svd(M_w.T)
            eta.append(s[1] / s[0])
        
            
            if eta[t-t_window] != 0:
                zeta_k = (eta[t] - eta[t-t_window]) / (eta[t-t_window] * (t_window))
                zeta.append(zeta_k)
            New_Batch = np.reshape(X_mat[t,:], (-1, 1))
            M_w = np.delete(M_w, (0), axis=0)
            M_w = np.row_stack((M_w, New_Batch.T))
            
        Zeta_m.append(zeta)
    Zeta_m = np.array(Zeta_m)
    print(Zeta_m.shape)
    return Zeta_m


##unwrapping func
def unwrap_angle(df1):
    #input df1 is a dataframe
    df=df1.copy()

    
    df=np.radians(df)
    x=df.sort_index()
    array1=x.iloc[:len(df)-1]
    array2=x.iloc[1:]    
    
    diff=np.array(array2)-np.array(array1)
    idx=np.where(abs(diff)>=np.pi)[0]
    
    for i in idx:
        #print(i)
        x[i+1:]=x[i+1:]+np.pi*2*-1*np.sign(diff[i])
    
    return(x)




def zz():


    a = rd_z(1)
    for i in range(2,5):
        a2 = rd_z(i)
        a =  np.append(a,a2,axis = 0)
    print(a.shape)    
    return a 
    
def plot_eve():
    df1 = raw(5)
    
    plt.figure( figsize=(20,16))
    plt.subplot(6,1,1)
    df2 = np.load("vpm.npy")
    plt.plot(range(df2.shape[1]),df2[0])
    plt.ylabel("Zeta of vp_m")  
    plt.xticks([0,round(5111940/8),5111940/4,round(3*5111940/8),
                5111940/2,round(5*5111940/8),round(6*5111940/8),round(7*5111940/8),5111940],
    ["0:00","3:00","6:00","9:00","12:00","15:00","18:00","21:00","24:00"])
    
    plt.subplot(6,1,2)
    df2 = np.load("vpa.npy")
    plt.plot(range(df2.shape[1]),df2[0])
    plt.ylabel("Zeta of vp_a")  
    plt.xticks([0,round(5111940/8),5111940/4,round(3*5111940/8),
                5111940/2,round(5*5111940/8),round(6*5111940/8),round(7*5111940/8),5111940],
    ["0:00","3:00","6:00","9:00","12:00","15:00","18:00","21:00","24:00"])
    plt.subplot(6,1,3)
    df2 = np.load("ipm.npy")
    plt.plot(range(df2.shape[1]),df2[0])
    plt.ylabel("Zeta of ip_m")  
    plt.xticks([0,round(5111940/8),5111940/4,round(3*5111940/8),
                5111940/2,round(5*5111940/8),round(6*5111940/8),round(7*5111940/8),5111940],
    ["0:00","3:00","6:00","9:00","12:00","15:00","18:00","21:00","24:00"])    
    
    plt.subplot(6,1,4)    
    df2 = np.load("ipa.npy")
    plt.plot(range(df2.shape[1]),df2[0])
    plt.ylabel("Zeta of ip_a")    
    
    plt.xticks([0,round(5111940/8),5111940/4,round(3*5111940/8),
                5111940/2,round(5*5111940/8),round(6*5111940/8),round(7*5111940/8),5111940],
    ["0:00","3:00","6:00","9:00","12:00","15:00","18:00","21:00","24:00"])
    
    plt.subplot(6,1,5)    
    df2 = np.load("f.npy")
    plt.plot(range(df2.shape[1]),df2[0]) 
    plt.ylabel("Zeta of frequency")    
    
    plt.xticks([0,round(5111940/8),5111940/4,round(3*5111940/8),
                5111940/2,round(5*5111940/8),round(6*5111940/8),round(7*5111940/8),5111940],
    ["0:00","3:00","6:00","9:00","12:00","15:00","18:00","21:00","24:00"])
    plt.grid(ls='--')
    plt.subplot(6,1,6)  
    for i in range(23):
        plt.plot(range(df1.shape[1]),df1[i])
    plt.xticks([0,round(5111940/8),5111940/4,round(3*5111940/8),
                5111940/2,round(5*5111940/8),round(6*5111940/8),round(7*5111940/8),5111940],
    ["0:00","3:00","6:00","9:00","12:00","15:00","18:00","21:00","24:00"])
    
    plt.ylabel("Rocof")
    # plt.xticks(range(df1.shape[1]),('Tom','Dick','Harry','Sally','Sue','Lily','Ava','Isla','Rose','Jack','Leo'))
    plt.grid(ls='--')
    # plt.title("event type: "+ca[int(pk3[j])])
    plt.tight_layout()             
    plt.savefig("rocof")  

def plot_orignal():
    plt.figure( figsize=(20,16))
    for j in range(6):

        df1 = rd_raw(2,j)
        plt.subplot(6,1,j+1)    
        for i in range(23):
            plt.plot(range(df1.shape[1]),df1[i])  
        if(j == 0):
            plt.ylim(524000, 549000)
    plt.grid(ls='--')
    plt.title("Mar_3")
    plt.tight_layout()             
    plt.savefig("Mar_3") 


def plot_zeta():
    word2 =["vpm","vpa","ipm","ipa","f","rocof"]
    plt.figure( figsize=(20,16))
    for i in range(6):
        df2 = np.load(word2[i]+".npy")
        plt.subplot(6,1,i+1)    
        # plt.plot(range(df1.shape[1]),df1[i]) 
        plt.plot(range(df2.shape[1]),df2[0])
    plt.grid(ls='--')
    plt.title("Mar_1z")
    plt.tight_layout()             
    plt.savefig("Mar_1z") 


    
def pp(a,j):
    b = np.zeros((a.shape[0],a.shape[1]))
    for i in range(a.shape[0]):
        if(j == 0):
        # print(np.mean(a[i]))
        # b[i] = np.deg2rad(a[i])
            b[i] = a[i]/np.mean(a[i])
        elif(j == 1):
            b[i] = np.deg2rad(a[i])
        elif(j == 2):
            b[i] = a[i]/100
        elif(j == 3):
            b[i] = np.deg2rad(a[i])
        elif(j == 4):
            b[i] = a[i]/np.mean(a[i])
        elif(j == 5):
            b[i] = a[i]
    return b

def read_shape(ii):
    p1 = open('X_S'+str(ii)+'.pickle',"rb")
    df1 = pickle.load(p1)        
    print(df1.shape)
    

def save_zeta(j):
    word2 =["vpm","vpa","ipm","ipa","f","rocof"]

    a = rd_raw(2,j)
    a = pp(a,j)
    c = get_single(a,30)
    np.save("Mar_3_"+word2[j]+'.npy',c) 
    print("allday saved")
    
def main():
    s1 = timeit.default_timer()  

    plot_orignal()
    # plot_zeta()
    s2 = timeit.default_timer()
    print('Time: ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

