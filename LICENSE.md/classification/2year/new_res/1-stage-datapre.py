

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
# 


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
    
def proces2(data,k):
    st1 =[]
    data = np.squeeze(data)
    
    for j in range(0,data.shape[0]):
        st2=[]
        for i in range(0,data.shape[1]):
            if(k == 0):
                temp = data[j,i,:]/np.mean(data[j,i,:])
            elif(k == 1):
                temp = data[j,i,:]/np.mean(data[j,i,:])
            elif(k == 2):
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
    # list = ['vp_m','vp_a','ip_m','ip_a','f','rocof']
    list = ['vp_m','ip_m','rocof']
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

    X_train = proces2(X_train,k) 

    return X_train,y_train


def rd_zeta(ii):
    path1 = '../zeta_all/30/'
    p1 = open(path1 +'X_S'+str(ii)+'.pickle',"rb")
    pk1 = pickle.load(p1)
    print(pk1.shape)
    return pk1
    
def rd_zeta2(ii):

    path1 = '../zeta_all/210/'
    p1 = open(path1 +'X_S'+str(ii)+'.pickle',"rb")
    pk1 = pickle.load(p1)
    print(pk1.shape)
    return pk1
    
def datapack(ii):
    k =0
    a,y = rd3(ii,k)
    for k in range(1,2+1):
        a2,_ = rd3(ii,k)
        a = np.concatenate((a, a2), axis=2)  

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
            

def get_info(x,j,feature,ind):
    up = []
    dn = []
    sup = []
    sdn = []
    dup =[]
    ddn =[]
    global reg
    for i in range(23):
        temp = x[j,i,feature,ind-reg:ind+reg]
        mm = np.mean(x[j,i,0, :])
        m1,m2 = cal_dif(temp)
        m3,m4 = cal_diparea(temp,mm)

        m5 = np.max(temp)-mm
        m6 = mm-np.min(temp)
        up.append(m5)
        dn.append(m6)
        
        sdn.append(m3)
        sup.append(m4)
        
        dup.append(m1)
        ddn.append(m2)

    return up,dn,dup,ddn,sup,sdn
    
    
def cal(ii):
    x,y = datapack(ii)
    x2 = rd_zeta(ii)
    x3 = rd_zeta2(ii)
    global reg
    v= 0
    st =[]
    label=[]
    # and (y[j,0] ==0 or y[j,0] ==4 )
    for j in range(x.shape[0]):
        # if(v<5 ):
        ind = np.argmax(x2[j,0])+reg+20
        ind2 = np.argmax(x3[j,0])+reg+20
        print(ind)
        list_v_up = []
        list_v_dn = []
        list_v_dup = []
        list_v_ddn = []       
        list_v_aup =[]
        list_v_adn =[]
        
        list_i_up = []
        list_i_dn = []
        list_i_dup = []
        list_i_ddn = []       
        list_i_aup =[]
        list_i_adn =[]

        list_r_up = []
        list_r_dn = []
        list_r_dup = []
        list_r_ddn = []       
        list_r_aup =[]
        list_r_adn =[]
        
        list_v_up,list_v_dn,list_v_dup,list_v_ddn,list_v_aup,list_v_adn = get_info(x,j,0,ind)
        list_i_up,list_i_dn,list_i_dup,list_i_ddn,list_i_aup,list_i_adn = get_info(x,j,1,ind)
        list_r_up,list_r_dn,list_r_dup,list_r_ddn,list_r_aup,list_r_adn = get_info(x,j,2,ind2)
        

        sd = []
        sd.append( np.max(list_v_up))
        sd.append( np.mean(list_v_up))
        sd.append( np.min(list_v_up))
        
        sd.append( np.max(list_v_dn))
        sd.append( np.mean(list_v_dn))
        sd.append( np.min(list_v_dn))
        
        sd.append( np.max(list_v_dup))
        sd.append( np.mean(list_v_dup))
        sd.append( np.min(list_v_dup))
        
        sd.append( np.max(list_v_ddn))
        sd.append( np.mean(list_v_ddn))
        sd.append( np.min(list_v_ddn))
        
        sd.append( np.max(list_v_aup))
        sd.append( np.mean(list_v_aup))
        sd.append( np.min(list_v_aup))
        
        sd.append( np.max(list_v_adn))
        sd.append( np.mean(list_v_adn))
        sd.append( np.min(list_v_adn))


        sd.append( np.max(list_i_up))
        sd.append( np.mean(list_i_up))
        sd.append( np.min(list_i_up))
        
        sd.append( np.max(list_i_dn))
        sd.append( np.mean(list_i_dn))
        sd.append( np.min(list_i_dn))
        
        sd.append( np.max(list_i_dup))
        sd.append( np.mean(list_i_dup))
        sd.append( np.min(list_i_dup))
        
        sd.append( np.max(list_i_ddn))
        sd.append( np.mean(list_i_ddn))
        sd.append( np.min(list_i_ddn))
        
        sd.append( np.max(list_i_aup))
        sd.append( np.mean(list_i_aup))
        sd.append( np.min(list_i_aup))
        
        sd.append( np.max(list_i_adn))
        sd.append( np.mean(list_i_adn))
        sd.append( np.min(list_i_adn))        


        sd.append( np.max(list_r_up))
        sd.append( np.mean(list_r_up))
        sd.append( np.min(list_r_up))
        
        sd.append( np.max(list_r_dn))
        sd.append( np.mean(list_r_dn))
        sd.append( np.min(list_r_dn))
        
        sd.append( np.max(list_r_dup))
        sd.append( np.mean(list_r_dup))
        sd.append( np.min(list_r_dup))
        
        sd.append( np.max(list_r_ddn))
        sd.append( np.mean(list_r_ddn))
        sd.append( np.min(list_r_ddn))
        
        sd.append( np.max(list_r_aup))
        sd.append( np.mean(list_r_aup))
        sd.append( np.min(list_r_aup))
        
        sd.append( np.max(list_r_adn))
        sd.append( np.mean(list_r_adn))
        sd.append( np.min(list_r_adn))       

        
        v+=1
        st.append(sd)
        label.append(y[j,0])
        
    st= np.array(st)
    label = np.array(label)
    df = pd.DataFrame(st)
    
    print(df.head())
    list1 = ["max","mean","min"]
    list_v_up,list_v_dn,list_v_dup,list_v_ddn,list_v_aup,list_v_adn
    list2 = ["v_up","v_dn","v_dup","v_ddn","v_aup","v_adn",
            "i_up","i_dn","i_dup","i_ddn","i_aup","i_adn",
            "r_up","r_dn","r_dup","r_ddn","r_aup","r_adn"
            ]
    list3 =[]
    
    for i in range(0,len(list2)):
        for j in range(0,len(list1)):
            list3.append(list1[j]+"_"+list2[i])      
    # print(list3)
    df.columns = list3
    df["label"] = label
    
    print(df.head())
    print(df.shape)
    
    df.to_csv("data/Ss"+str(ii)+".csv", index = None)
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
    sum1 =0
    sum2 =0
    for i in range(temp.shape[0]):
        if(temp[i]<mm):
            sum1+= mm-temp[i]
        elif (temp[i]>mm):
            sum2+= temp[i] - mm
    return sum1,sum2
    
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
    

def remove_z(str):
    if str[0] == '0':
        return str[1]
    else:
        return str
        
        
def add_der(ii):
    st =[]
    path1 = '../../../pickleset2/'
    list = ['vp_m','vp_a','ip_m','ip_a','f','rocof']
    df = pd.read_csv("data/Ss"+str(ii)+".csv")
    
    dt  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[0])+'_6.csv')
    df["word"] = dt["1"]
    df["word2"] = dt["2"]
    df.to_csv("data/Ss"+str(ii)+".csv",index= None)
def fuu(ii):
    st =[]
    df = pd.read_csv("data/Ss"+str(ii)+".csv")
    dt = df["word2"].values
    for i in range(df.shape[0]):
        temp = dt[i].split("_")
        st.append(temp[0]+"_"+temp[1]+"_"+remove_z(temp[2])+"_"+temp[3])
    st = np.array(st)
    df["word2"] = st
    df.to_csv("data/Ss"+str(ii)+".csv",index= None)

def add_order(ii):
    st =[]
    df = pd.read_csv("data/Ss"+str(ii)+".csv")
    df["season"] = ii
    ll = list(range(df.shape[0]))
    ll = np.array(ll)
    # for i in range():
        
    df["No"] = ll
    df.to_csv("data/Ss"+str(ii)+".csv",index= None)
reg = 5*60
def main():
    s1 = timeit.default_timer()  
    
    # test()
    # cal(1)
    # pack()
    for ii in range(1,13+1):
        cal(ii)
        add_der(ii)
        fuu(ii)
        add_order(ii)
    s2 = timeit.default_timer()
    print('Time:(min) ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

