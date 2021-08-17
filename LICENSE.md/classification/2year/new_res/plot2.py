

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



def proces3(data,k):
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
                temp = data[j,i,:]/np.mean(data[j,i,:])            
            elif(k == 3):
                temp = np.deg2rad(data[j,i,:])                
            elif(k == 4):
                temp = data[j,i,:]/60          
            elif(k == 5):
                temp = data[j,i,:]      
            st2.append(temp)
        st1.append(st2)

    st1 = np.array(st1)
    st1 = st1[:,:,np.newaxis,:]
    # print(st1.shape)
    return st1
    
    
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

# word =["vpm","vpa","ipm","ipa","freq","Rocof","Active","Reactive"]



    
def rd2(ii,k):
    path1 = '../../../pickleset2/'
    list = ['vp_m','ip_m','vp_a','ip_a','f']

    p1 = open(path1 +'X_S'+str(ii)+'_'+str(list[k])+'_6.pickle',"rb")

    pk3  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[k])+'_6.csv')
   
    pk1 = pickle.load(p1)

    X_train = pk1
    y_train = pk3.values
    
    path2 = "../rm_index/"


    tr = np.load(path2+'S'+str(ii)+'.npy') 

    # X_train = X_train[tr]
    # y_train = y_train[tr]
    X_train=X_train.transpose(0,1,3,2)
    # X_train = proces(X_train,k) 
    print(X_train.shape)
    print(y_train.shape)
    return X_train, y_train        

def rd3(ii,k):

    path1 = '../../../pickleset2/'
    res = ['vp_m','vp_a','ip_m','ip_a','f','rocof']

    p1 = open(path1 +'X_S'+str(ii)+'_'+str(res[k])+'_6.pickle',"rb")
    pk1 = pickle.load(p1)
    
    df  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(res[k])+'_6.csv')
   

    # df  = pd.read_csv(path1 +'S'+str(ii)+'.csv')
    dt = df.loc[df["1"].str.contains("Transformer_Trip|Transformer_Lightning|Transformer_Planned")].index
    print(dt.shape)
    dt2 = df.loc[df["1"].str.contains("Line_Trip|Line_Lightning")==False ].index
    # df[df["col1"].str.contains('this|that')==False
    print(dt2.shape)
    
    path2 = "../../../../../"
    dg = pd.read_csv(path2 +'muti.csv')
    print("dg.shape",dg.shape)
    dg = dg[(dg["v"]==4) | (dg["v"]==7)]
    ll = dg["new"].values.tolist()
    
    dt3 = df.loc[df["2"].isin(ll)].index
    print(dt3.shape)
    ind1 = list(set(dt)^set(dt2))
    ind2 = list(set(dt)^set(dt3))
    df.loc[dt2,("0")] = 0
    df.loc[dt,("0")] = 1
    df.loc[ind2,("0")] = 1
    # df["0"][dt2] = 0   
    # df["0"][dt] = 1
    # df["0"][ind2] = 1    
    
    X_train = pk1
    y_train = df.values
    
    path2 = "../rm_index/"

    tr = np.load(path2+'S'+str(ii)+'.npy') 

    # X_train = X_train[tr]
    # y_train = y_train[tr]

    X_train=X_train.transpose(0,1,3,2)

    X_train = proces3(X_train,k) 

    return X_train,y_train


def rd_power(ii):
    X_vpm, y_vpm  = rd2(ii,0)
    X_ipm, y_ipm  = rd2(ii,1)
    X_vpa, y_vpa  = rd2(ii,2)
    X_ipa, y_ipa  = rd2(ii,3)

    
    temp3 =[]
    for i in range(X_vpm.shape[2]):
        temp3.append(4*np.pi*i)

    temp3 = np.array(temp3)
    temp3 = temp3[:,np.newaxis]
    # print("temp3.shape",temp3.shape)
    st1 =[]
    st3 =[]
    # y_vpm.shape[0]
    for j in range(y_vpm.shape[0]):
        st2 = []
        st4 = []
        for i in range(0,23):
            # print("X_vpa[j,i,:,:].shape",X_vpa[j,i,:,:].shape)
            # temp1 = math.sqrt(3)*X_vpm[j,i,:,:]*X_ipm[j,i,:,:]*np.cos(np.deg2rad(X_vpa[j,i,:,:]-X_ipa[j,i,:,:]))/1000000
            # temp2 = math.sqrt(3)*X_vpm[j,i,:,:]*X_ipm[j,i,:,:]*np.sin(np.deg2rad(X_vpa[j,i,:,:]-X_ipa[j,i,:,:]))/1000000
            temp1 = math.sqrt(3)*X_vpm[j,i,:,:]*X_ipm[j,i,:,:]*np.cos(np.deg2rad(X_vpa[j,i,:,:]-X_ipa[j,i,:,:]))
            temp2 = math.sqrt(3)*X_vpm[j,i,:,:]*X_ipm[j,i,:,:]*np.sin(np.deg2rad(X_vpa[j,i,:,:]-X_ipa[j,i,:,:]))           
            # print("temp1.shape",temp1.shape)
            st2.append(temp1/np.mean(temp1))
            st4.append(temp2/np.mean(temp2))
        st1.append(st2)
        st3.append(st4)
    st1 = np.array(st1)
    st3 = np.array(st3)
    # print(st1.shape)
    # print(st3.shape)
    return st1,st3
    
def rd_zeta(ii):
    path1 = '../zeta_all/30/'
    # path1 = '../zeta/150/'
    p1 = open(path1 +'X_S'+str(ii)+'.pickle',"rb")
    pk1 = pickle.load(p1)
    print(pk1.shape)
    return pk1
    
def rd_zeta2(ii):
    # path1 = '../zeta_all/30/'
    path1 = '../zeta_all/150/'
    p1 = open(path1 +'X_S'+str(ii)+'.pickle',"rb")
    pk1 = pickle.load(p1)

    print(pk1.shape)
    return pk1
        
    
    
def datapack(ii):
    k =0
    a,y = rd3(ii,k)
    for k in range(1,5+1):
        a2,_ = rd3(ii,k)
        a = np.concatenate((a, a2), axis=2)  
    # st1,st2 = rd_power(ii)
    # a = np.concatenate((a, st1), axis=2) 
    # a = np.concatenate((a, st2), axis=2) 
    print(a.shape)
    print(y.shape)
    return a,y
    
    
def plot_f(ii):
    x,y = datapack(ii)
    x2 = rd_zeta(ii)
    # x.shape[0]
    word = ["vp_m","vp_a","ip_m","ip_a",
            "zeta_vp_m","zeta_vp_a","zeta_ip_m","zeta_ip_a",
            "freq","rocof","actPow","reaPow",
            "zeta_freq","zeta_rocof","zeta_actPow","zeta_reaPow"
            ]
    v= 0
    # 
    for j in range(x.shape[0]):
        # if(v<10 and (y[j,1] ==2 )):
        # if(v<10 and y[j,0] == 0):
        if(v<10 and ("Freq"  in y[j,1])):
            print(j)
            plt.figure( figsize=(20,8))
            for i in range(4):
                plt.subplot(4,4,i+1)
                for i2 in range(23):
                    plt.plot(range(x.shape[3]),x[j,i2,i,:])
                plt.ylabel(word[i])
            for i in range(4,8):
                plt.subplot(4,4,i+1)
                # for i2 in range(23):
                plt.plot(range(x2.shape[2]),x2[j,i-4,:])
                plt.ylabel(word[i])    
            for i in range(8,12):
                plt.subplot(4,4,i+1)
                for i2 in range(23):
                    plt.plot(range(x.shape[3]),x[j,i2,i-4,:])
                plt.ylabel(word[i])
                plt.xlabel(np.argmin(x[j,0,i-4,:]))
            for i in range(12,16):
                if(i == 13):
                    plt.subplot(4,4,i+1)
                    # for i2 in range(23):
                    plt.plot(range(x2.shape[2]),x2[j,i-9,:])                
                    plt.ylabel(word[i-1])
                    plt.xlabel(np.argmax(x2[j,i-9,:]))
                else:
                    plt.subplot(4,4,i+1)
                    # for i2 in range(23):
                    plt.plot(range(x2.shape[2]),x2[j,i-8,:])                
                    plt.ylabel(word[i])
            plt.title("Event : "+str(y[j,1])+" " +str(y[j,2]))
            plt.grid(ls='--')
            plt.tight_layout()             
            plt.savefig("S_"+str(ii)+"_"+str(j))                  
            v+=1
            
            
def plot_s(ii):
    x,y = datapack(ii)
    # x2 = rd_zeta(ii)
    # num = 27
    # ll = ["2016-2-7 7:20","2016-2-7 7:21","2016-2-7 7:22"]
    
    # num = 11
    # ll = ["2016-2-2 19:39","2016-2-2 19:40","2016-2-2 19:41"]    
    word = ["vp_m","vp_a","ip_m","ip_a",
            "zeta_vp_m","zeta_vp_a","zeta_ip_m","zeta_ip_a",
            "freq","rocof","actPow","reaPow",
            "zeta_freq","zeta_rocof","zeta_actPow","zeta_reaPow"
            ]    
    cc = range(0,10800,int(10800/3))
    
    for j in range(num,num+1):
        plt.figure( figsize=(10,4))

        for i2 in range(23):
            plt.plot(range(x.shape[3]),x[j,i2,0,:])
        plt.ylabel(word[0])

        plt.title("Event : "+str(y[j,1])+" " +str(y[j,2]))
        plt.grid(ls='--')
        plt.tight_layout() 
        plt.xticks(cc,ll)         
        plt.savefig("S_"+str(ii)+"_"+str(j))                  
        # v+=1
        
        
def plot_s2(ii):
    x,y = datapack(ii)
    # x2 = rd_zeta(ii)
    # num = 53
    # ll = ["2016-2-10 19:41","2016-2-10 19:42","2016-2-10 19:43"]
    
    # num = 30
    # ll = ["2016-2-8 10:14","2016-2-8 10:15","2016-2-8 10:16"]    
    # word = ["vp_m","vp_a","ip_m","ip_a",
            # "zeta_vp_m","zeta_vp_a","zeta_ip_m","zeta_ip_a",
            # "freq","rocof","actPow","reaPow",
            # "zeta_freq","zeta_rocof","zeta_actPow","zeta_reaPow"
            # ]    
    cc = range(0,10800,int(10800/3))
    # ll2 =[3,5,22,53,54,84,89,108,109]
    for j in range(num,num+1):
    # for j in ll2:
        plt.figure( figsize=(10,4))

        for i2 in range(23):
            plt.plot(range(x.shape[3]),x[j,i2,5,:])
        plt.ylabel("rocof")

        plt.title("Event : "+str(y[j,1])+" " +str(y[j,2]))
        plt.grid(ls='--')
        plt.tight_layout() 
        plt.xticks(cc,ll)         
        plt.savefig("S_"+str(ii)+"_"+str(j))                  
        # v+=1
        
def plot_s3(ii):
    x,y = datapack(ii)
    x2 = rd_zeta(ii)
    x3 = rd_zeta2(ii)
    cc = range(0,10800,int(10800/3))
    # ll2 =[3,5,22,53,54,84,89,108,109]
    num = 53
    ll = ["2016-2-10 19:41","2016-2-10 19:42","2016-2-10 19:43"]
    word = ["vp_m","vp_a","ip_m","ip_a",
            "zeta_vp_m","zeta_vp_a","zeta_ip_m","zeta_ip_a",
            "freq","rocof","actPow","reaPow",
            "zeta_freq","zeta_rocof","zeta_actPow","zeta_reaPow"
            ]    
    
    
    for j in range(num,num+1):
    # for j in ll2:
        plt.figure( figsize=(10,9))
        plt.subplot(3,1,1)
        for i2 in range(23):
            plt.plot(range(x.shape[3]),x[j,i2,5,:])
            # ${m_2}$
        plt.ylabel("${Rof_i}$(t)")
        plt.title("Event : "+str(y[j,1])+" " +str(y[j,2]))
        plt.grid(ls='--')
        plt.xticks(cc,ll)
        
        plt.subplot(3,1,3)
        # for i2 in range(23):
        plt.plot(range(x2.shape[2]),x2[j,0,:])
        plt.xlabel("Window size = 30")
        # plt.title("Event : "+str(y[j,1])+" " +str(y[j,2]))
        plt.grid(ls='--')
        # plt.xticks(cc,ll)
        plt.ylabel(r'$\xi$'+"(30,t)")
        
        plt.subplot(3,1,2)
        # for i2 in range(23):
        plt.ylabel(r'$\xi$'+"(150,t)")
        plt.plot(range(x3.shape[2]),x3[j,0,:])
        plt.xlabel("Window size = 150")
        
        # plt.title("Event : "+str(y[j,1])+" " +str(y[j,2]))
        plt.grid(ls='--')
        plt.tight_layout() 
        
        # plt.ylabel("$\xi$(150,t)")
        # plt.xticks(cc,ll)         
        plt.savefig("S_"+str(ii)+"_"+str(j))   
        
def main():
    s1 = timeit.default_timer()  
    for ii in range(1,1+1):
        plot_s3(ii)
    # rd2(1,1)
    s2 = timeit.default_timer()
    print('Time:(min) ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

