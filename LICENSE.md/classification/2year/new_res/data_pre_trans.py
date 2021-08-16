

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

word =["vpm","vpa","ipm","ipa","freq","Rocof","Active","Reactive"]


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
    list = ['vp_m','vp_a','ip_m','ip_a','f','rocof']

    p1 = open(path1 +'X_S'+str(ii)+'_'+str(list[k])+'_6.pickle',"rb")
    pk1 = pickle.load(p1)
    
    pk3  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[k])+'_6.csv')
   


    X_train = pk1
    y_train = pk3.values
    
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
    
    # X_vpm, y_vpm  = rd3(ii,0)
    # X_ipm, y_ipm  = rd3(ii,1)
    # X_vpa, y_vpa  = rd3(ii,2)
    # X_ipa, y_ipa  = rd3(ii,3)    
    
    
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
    # path1 = '../zeta/30/'
    # path1 = '../zeta/150/'
    path1 = '../zeta_all/30/'
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
    st1,st2 = rd_power(ii)
    a = np.concatenate((a, st1), axis=2) 
    a = np.concatenate((a, st2), axis=2) 
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
        if(v<10 and (y[j,0] ==0 or y[j,0] ==4 )):
        # if(v<10 and y[j,0] == 0):
        # if(y[j,0] == 2):
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
            
            
def cal(ii):
    x,y = datapack(ii)
    x2 = rd_zeta(ii)
    # x.shape[0]
    word = ["vp_m","vp_a","ip_m","ip_a",
            "zeta_vp_m","zeta_vp_a","zeta_ip_m","zeta_ip_a",
            "freq","rocof","actPow","reaPow",
            "zeta_freq","zeta_rocof","zeta_actPow","zeta_reaPow"
            ]
    v= 0
    st =[]
    label=[]
    # and (y[j,0] ==0 or y[j,0] ==4 )
    for j in range(x.shape[0]):
        # if(v<5 ):
        ind = np.argmax(x2[j,0])+200
        print(ind)
        list_v_up = []
        list_v_dn = []
        list_v_a =[]
        
        list_i_up = []
        list_i_dn = []
        list_i_a =[]
        
        list_p_up = []
        list_p_dn = []
        list_p_a =[]
        
        list_q_up = []
        list_q_dn = []
        list_q_a =[]
        
        for i in range(23):
            temp = x[j,i,0,ind-180:ind+180]
            mm = np.mean(x[j,i,0, :])
            m1,m2 = cal_dif(temp)
            m3 = cal_diparea(temp,mm)
            list_v_dn.append(m1)
            list_v_up.append(m2)
            list_v_a.append(m3)
            
            temp = x[j,i,2,ind-180:ind+180]
            mm = np.mean(x[j,i,2, :])
            m1,m2 = cal_dif(temp)
            m3 = cal_diparea(temp,mm)
            list_i_dn.append(m1)
            list_i_up.append(m2)
            list_i_a.append(m3)
            
            
            temp = x[j,i,6,ind-180:ind+180]
            mm = np.mean(x[j,i,6, :])
            m3 = cal_diparea(temp,mm)
            m1,m2 = cal_dif(temp)     
            list_p_dn.append(m1)
            list_p_up.append(m2)
            list_p_a.append(m3)
            
            temp = x[j,i,7,ind-180:ind+180]
            mm = np.mean(x[j,i,7, :])
            m3 = cal_diparea(temp,mm)
            m1,m2 = cal_dif(temp)     
            list_q_dn.append(m1)
            list_q_up.append(m2)
            list_q_a.append(m3)
            
        kk = np.zeros(36)
        kk[0] = np.max(list_v_dn)
        kk[1] = np.mean(list_v_dn)
        kk[2] = np.min(list_v_dn)
        
        kk[3] = np.max(list_v_up)
        kk[4] = np.mean(list_v_up)
        kk[5] = np.min(list_v_up)

        kk[6] = np.max(list_i_dn)
        kk[7] = np.mean(list_i_dn)
        kk[8] = np.min(list_i_dn)
        
        kk[9] = np.max(list_i_up)
        kk[10] = np.mean(list_i_up)
        kk[11] = np.min(list_i_up)

        kk[12] = np.max(list_p_dn)
        kk[13] = np.mean(list_p_dn)
        kk[14] = np.min(list_p_dn)
        
        kk[15] = np.max(list_p_up)
        kk[16] = np.mean(list_p_up)
        kk[17] = np.min(list_p_up)

        kk[18] = np.max(list_q_dn)
        kk[19] = np.mean(list_q_dn)
        kk[20] = np.min(list_q_dn)
        
        kk[21] = np.max(list_q_up)
        kk[22] = np.mean(list_q_up)
        kk[23] = np.min(list_q_up)
        
        kk[24] = np.max(list_v_a)
        kk[25] = np.mean(list_v_a)
        kk[26] = np.min(list_v_a)       

        kk[27] = np.max(list_i_a)
        kk[28] = np.mean(list_i_a)
        kk[29] = np.min(list_i_a)  


        kk[30] = np.max(list_p_a)
        kk[31] = np.mean(list_p_a)
        kk[32] = np.min(list_p_a)   

        kk[33] = np.max(list_q_a)
        kk[34] = np.mean(list_q_a)
        kk[35] = np.min(list_q_a)   
        
        v+=1
        st.append(kk)
        label.append(y[j,0])
        
    st= np.array(st)
    label = np.array(label)
    df = pd.DataFrame(st)
    
    print(df.head())
    list1 = ["max","mean","min"]
    list2 = ["v_dn","v_up","i_dn","i_up","p_dn","p_up","q_dn","q_up","v_a","i_a","p_a","q_a"]
    list3 =[]
    
    for i in range(0,len(list2)):
        for j in range(0,len(list1)):
            list3.append(list1[j]+"_"+list2[i])      
    # print(list3)
    df.columns = list3
    df["label"] = label
    
    print(df.head())
    print(df.shape)
    
    df.to_csv("data/S"+str(ii)+".csv", index = None)
    print("S"+str(ii)+" save done")
def test():
    list1 = ["max","mean","min"]
    list2 = ["v_dn","v_up","i_dn","i_up","p_dn","p_up","q_dn","q_up","v_a","i_a","p_a","q_a"]
    list3 =[]
    
    for i in range(0,len(list2)):
        for j in range(0,len(list1)):
            list3.append(list1[j]+"_"+list2[i])      
    print(list3)
    
def cal_diparea(temp,mm):
    sum =0
    for i in range(temp.shape[0]):
        if(temp[i]<mm):
            sum+= mm-temp[i]
    return sum
    
def cal_dif(temp):
    d = int(temp.shape[0]/2)
    m1 = 0
    m2 = 0
    for i in range(d):
        temp1 = temp[i]-temp[i+d]
        temp2 = temp[i+d]-temp[i]
        if(temp1>m1):
            m1 = temp1
        if(temp2>m2):
            m2 = temp2
    # print(m1,m2)
    return m1,m2
    
def merge_label(ii):
    path1 = '../../../pickleset2/'
    list = ['vp_m','vp_a','ip_m','ip_a','f','rocof']
    k =1 
    pk3  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[k])+'_6.csv')
    # pk3 = pk3[""]
    df = pd.read_csv("data/S"+str(ii)+".csv")
    df["word2"] = pk3["2"]
    print(df.head())
    print(df.shape)
    df.to_csv("data/S"+str(ii)+".csv",index= None)

def remove_z(str):
    if str[0] == '0':
        return str[1]
    else:
        return str
def fuu(ii):
    st =[]
    df = pd.read_csv("data/S"+str(ii)+".csv")
    dt = df["word2"].values
    for i in range(df.shape[0]):
        temp = dt[i].split("_")
        st.append(temp[0]+"_"+temp[1]+"_"+remove_z(temp[2])+"_"+temp[3])
    st = np.array(st)
    df["word2"] = st
    df.to_csv("data/S"+str(ii)+".csv",index= None)
    
def pack():
    ii = 1
    df = pd.read_csv("data/S"+str(ii)+".csv")
    for ii in range(2,14):
        df2 = pd.read_csv("data/S"+str(ii)+".csv")
        df = pd.concat([df,df2])
    df.to_csv("data/all.csv",index= None)
    
def main():
    s1 = timeit.default_timer()  
    # test()
    # cal(1)
    # pack()
    for ii in range(1,13+1):
        fuu(ii)
        # merge_label(ii)
    s2 = timeit.default_timer()
    print('Time:(min) ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

