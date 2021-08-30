
# Have to run in spam2 

import numpy as np
from scipy.stats import pearsonr
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
import heapq
from MatComp_Dyn import deal_miss
from ED_OLAP import get_single_feature
from random import randint
# import math
# 

def remove_z(str):
    if str[0] == '0':
        return str[1]
    else:
        return str
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
            elif(k == 3):
                temp = data[j,i,:]/60    
            st2.append(temp)
        st1.append(st2)

    st1 = np.array(st1)
    st1 = st1[:,:,np.newaxis,:]
    print(st1.shape)
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



def rd3(ii,k):

    path1 = '../../../pickleset2/'

    list = ['vp_m','ip_m','rocof','f']
    p1 = open(path1 +'X_S'+str(ii)+'_'+str(list[k])+'_6.pickle',"rb")
    pk1 = pickle.load(p1)
    
    df  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[k])+'_6.csv')
    df.columns = ["label","word","word1"]
    dt = df["word1"].values
    st =[]
    for i in range(df.shape[0]):
        temp = dt[i].split("_")
        st.append(temp[0]+"_"+temp[1]+"_"+remove_z(temp[2])+"_"+temp[3])
    st = np.array(st)
    df["word2"] = st
    
    # print(st[:2])
    dg = pd.read_csv('data/X_val.csv')

    ll = dg["word2"].values.tolist()
    dt1 = df.loc[df.word2.isin(ll)].index
    X_train = pk1[dt1]
    # print(X_train.shape)
    X_train=X_train.transpose(0,1,3,2)
    X_train = proces2(X_train,k) 
    y_train = df.loc[dt1,"label"]
    df  = df.loc[dt1]
    a = y_train.shape[0]
    print(X_train.shape)
    print(y_train.shape)
    # df.to_csv("data/y_k="+str(k)+"_"+str(ii)+".csv",index = 0)
    return X_train,y_train
    
def get_label(ii):

    path1 = '../../../pickleset2/'
    k = 0
    list = ['vp_m','ip_m','rocof','f']
    p1 = open(path1 +'X_S'+str(ii)+'_'+str(list[k])+'_6.pickle',"rb")
    pk1 = pickle.load(p1)
    
    df  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[k])+'_6.csv')
    df.columns = ["label","word","word1"]
    dt = df["word1"].values
    st =[]
    for i in range(df.shape[0]):
        temp = dt[i].split("_")
        st.append(temp[0]+"_"+temp[1]+"_"+remove_z(temp[2])+"_"+temp[3])
    st = np.array(st)
    df["word2"] = st
    
    # print(st[:2])
    dg = pd.read_csv('data/X_val.csv')

    ll = dg["word2"].values.tolist()
    dt1 = df.loc[df.word2.isin(ll)].index
    y_train = df.loc[dt1]
    st2 = []
    st3 = []
    for i in range(df.shape[0]):
        temp = df.loc[i,"word2"]
        dt = dg[dg["word2"]==temp]
        if(dt.shape[0]>0):
            st2.append(dt["word2"].values)
            st3.append(dt["label"].values)
            
    print(len(st3))        
    print(len(st2))
    df = pd.DataFrame(st2)
    st3 = np.array(st3)
    df.columns = ["word2"]
    df["label"] = st3
    print(df.head())
    df.to_csv("data/y_"+str(ii)+".csv",index = 0)
    
    
    
    

def save_noise(ii,mis_rate):
    a,y = datapack(ii)

    st1 =[]
    st2 =[]
    st3 =[]
    # a.shape[0]
    for i in range(a.shape[0]):
        # 10800
        b1= deal_miss(a[i],int(10800/60),mis_rate)
        st1.append(b1)

    st1=np.array(st1)
    # st2=np.array(st2)
    # st3=np.array(st3)
    # features = {st1,st2,st3}
    ll = ["knn","sf","sfnb"]
    # for i in range(len(ll)):
    pickle_out = open("no_data/X"+str(ll[1])+"_"+str(mis_rate)+"_"+str(ii)+".pickle","wb")
    pickle.dump(st1, pickle_out, protocol=2)
    pickle_out.close()
    print("st1",st1.shape)
    print("save done")


def read_Xknn(ii):
    p1 = open("no_data/Xknn_0.1_"+str(ii)+'.pickle',"rb")
    a = pickle.load(p1)
    # _,y = rd3(ii,0)
    

    print(a.shape)
    # print(y.shape)
    return a
    
def read_sf(ii):
    p1 = open("no_data/Xsf_0.1_"+str(ii)+'.pickle',"rb")
    a = pickle.load(p1)

    print(a.shape)
    return a

def datapack(ii):
    k =0
    a,y = rd3(ii,k)
    for k in range(1,3+1):
        a2,_ = rd3(ii,k)
        a = np.concatenate((a, a2), axis=2)  

    # print(a.shape)
    # print(y.shape)
    return a,y


def rd_dp2(ii,k):

    path1 = '../../../pickleset2/'
    # list = ['vp_m','vp_a','ip_m','ip_a','f','rocof']
    list = ['vp_m','ip_m','rocof']
    p1 = open(path1 +'X_S'+str(ii)+'_'+str(list[k])+'_6.pickle',"rb")
    pk1 = pickle.load(p1)
    
    pk3  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[k])+'_6.csv')

    X_train = pk1
    y_train = pk3.values


    X_train=X_train.transpose(0,1,3,2)

    X_train = proces2(X_train,k) 
    print(X_train.shape)
    return X_train,y_train



def dp2(ii):
    k =0
    a,y = rd_dp2(ii,k)
    for k in range(1,2+1):
        a2,_ = rd_dp2(ii,k)
        a = np.concatenate((a, a2), axis=2)  

    print(a.shape)
    print(y.shape)
    return a,y
    

def rd_zeta(ii):
    path1 = 'zeta_miss/30/'
    # knn_S0.1_1
    # sf_S0.1
    
    p1 = open(path1 +'knn_S0.1_'+str(ii)+'.pickle',"rb")
    pk1 = pickle.load(p1)
    print(pk1.shape)
    return pk1
    
def rd_zeta2(ii):

    path1 = 'zeta_miss/150/'
    p1 = open(path1 +'knn_S0.1_'+str(ii)+'.pickle',"rb")
    pk1 = pickle.load(p1)
    print(pk1.shape)
    return pk1
   
def rd_z1(ii):
    path1 = '../zeta_all/30/'
    p1 = open(path1 +'X_S'+str(ii)+'.pickle',"rb")
    pk1 = pickle.load(p1)
    print(pk1.shape)
    return pk1
    
def rd_z2(ii):

    path1 = '../zeta_all/210/'
    p1 = open(path1 +'X_S'+str(ii)+'.pickle',"rb")
    pk1 = pickle.load(p1)
    print(pk1.shape)
    return pk1
    
   
   
def pre_olap1(ii,rate):  
    ss = 150

    path1 = "no_data/"
    ll = ["knn","sf","sfnb"]
    j = 1
    p1 = open(path1 +'X'+ll[j]+'_'+str(rate)+'_'+str(ii)+'.pickle',"rb")
    a = pickle.load(p1)
    print("a.shape",a.shape)

    st = [] 
        # 
    for i in range(a.shape[0]):
        print("i=",i)
        temp = a[i,:,3,:]
        temp = temp[:,np.newaxis,:]
        temp = get_single_feature(temp,ss)


        st.append(temp)
    X_train =np.array(st)
    print("X_train.shape",X_train.shape)    
    path3 = 'zeta_miss/'+str(ss)+'/'
    # save features for all the events
    
    pickle_out = open(path3 +ll[j]+"_S"+str(rate)+"_"+str(ii)+".pickle","wb")
    pickle.dump(X_train, pickle_out, protocol=2)
    pickle_out.close()    
    print("save done S"+str(ii))

def pre_olap2(ii,rate):  
    ss = 30

    path1 = "no_data/"
    ll = ["knn","sf","sfnb"]
    j = 1
    p1 = open(path1 +'X'+ll[j]+'_'+str(rate)+'_'+str(ii)+'.pickle',"rb")
    a = pickle.load(p1)
    
    print("a.shape",a.shape)
    

    st = [] 

    for i in range(a.shape[0]):
        print("i=",i)
        temp = a[i,:,0,:]
        temp = temp[:,np.newaxis,:]
        temp = get_single_feature(temp,ss)

        st.append(temp)
    X_train =np.array(st)
    print("X_train.shape",X_train.shape)    
    path3 = 'zeta_miss/'+str(ss)+'/'
    # save features for all the events
    
    pickle_out = open(path3 +ll[j]+"_S"+str(rate)+"_"+str(ii)+".pickle","wb")
    pickle.dump(X_train, pickle_out, protocol=2)
    pickle_out.close()    
    print("save done S"+str(ii))

    
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
def pearsonrSim(x,y):
    '''
    皮尔森相似度
    '''
    return pearsonr(x,y)[0]

def choose(X):

    nums = []
    for i in range(0,23):
        temp = X[i,:]
        nums.append(np.max(temp)-np.min(temp))
    # nums.sort()
    # print(nums)
    max_num_index_list = map(nums.index, heapq.nlargest(1, nums))

    ll = list(max_num_index_list)
    print("Largest dif ref:",ll[0])
    return np.squeeze(X[ll[0],:]),ll[0]
    
def cal(ii):
    x,y = dp2(ii)
    x2 = rd_z1(ii)
    x3 = rd_z2(ii)
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
    
    
def cal2(ii):
    x = read_Xknn(ii)
    # x,y = datapack(ii)
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
        # sd.append( np.min(list_sim))
        
        v+=1
        st.append(sd)
        # label.append(y[j])
        
    st= np.array(st)
    # label = np.array(label)
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
    # list3.append("sim")
    df.columns = list3
    # df["label"] = label
    
    # df["r1"] = df["mean_r_aup"]/df["mean_r_adn"]
    # df["r2"] = df["mean_r_aup"]/(df["mean_r_adn"]+df["mean_r_aup"])
    # df["r3"] = df["mean_r_up"]/df["mean_r_dn"]
    
    # df = df.replace([np.inf, -np.inf], np.nan)
    # ind = df[df.isnull().T.any()].index
    # df.loc[ind,:] = -1
    
    print(df.head())
    print(df.shape)
    
    df.to_csv("data/Sm"+str(ii)+".csv", index = None)
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
    # datapack(1)
    # get_label(1)
    # rd3(1,1)
    # sum = 0
    # for i in range(1,14):
        # _,_, = rd3(i,1)
        # sum+=c

    # pre_olap1(1)
    # save_noise(1)
    # rd3(1,1)
    # test()
    # cal(1)
    # pack()
    # pre_olap2(13,0.3)
    for ii in range(10,13+1):
        # pre_olap2(ii,0.1)
        # get_label(ii)
        # cal2(ii)
        # cal(ii)
        # add_der(ii)
        # fuu(ii)
        # add_order(ii)
        save_noise(ii,0.4)
        # rd3(ii,1)
        # read_Xknn(ii)
        # pre_olap1(ii,0.1)
        # pre_olap2(ii,0.1)
        # save_noise(ii,0.2)
        # save_noise(ii,0.05)
        # save_noise(ii,0.3)
        # save_noise(ii,0.4)
        # save_noise(ii,0.5)
        # pre_olap1(ii,0.1)
        # pre_olap1(ii,0.2)


        # cal2(ii)

    s2 = timeit.default_timer()
    print('Time:(min) ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

