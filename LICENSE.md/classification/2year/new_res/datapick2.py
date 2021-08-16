
# This is for Feq CWT^2 use
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import pandas as pd
import pywt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import heapq
from sklearn.model_selection import learning_curve, GridSearchCV  
from sklearn.svm import SVC    
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
import pywt
import pickle
import timeit
import datetime
from pyts.image import MarkovTransitionField
from random import shuffle
import collections
from scipy import signal 
import os
import sys
from scipy.linalg import svd 
import collections

start = timeit.default_timer()

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
    df2 = pd.read_csv("../../../../../B_T.csv")
    # df2 = df["new"].values.tolist()
    # df2 = df["Cause"].values
    dt = df2[df2["Cause"] == "Trip"]

    dt = dt["new"].values.tolist()
    # print(len(df))
    # print(df[:5])
    for i in range(len(y)):
        #print(i)
        temp = y[i,2].split("_")
        ww = temp[0]+"_"+temp[1]+"_"+remove_z(temp[2])+"_"+temp[3]
        # print(np.isin(ww,df))
        # and np.isin(ww,dt)==False
        if(np.isin(ww,df)==False and np.isin(ww,dt)==False):
        
            if y[i,0]==0:
                y_new.append(0)
                X_new.append(X[i,:,:,:])

                
                
            elif y[i,0]==1:
                y_new.append(1)
                X_new.append(X[i,:,:,:])

        
                
            elif y[i,0]==2:
                y_new.append(2)
                X_new.append(X[i,:,:,:])

            
            elif y[i,0]==3:
                y_new.append(3)
                X_new.append(X[i,:,:,:])

            elif y[i,0]==4:
                y_new.append(0)
                X_new.append(X[i,:,:,:])
  
            elif y[i,0]==5:
                y_new.append(1)
                X_new.append(X[i,:,:,:])

            
        
        
    return  np.array(X_new), np.array(y_new)
    # ,np.array(word)



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

    X_train = X_train[tr]
    y_train = y_train[tr]
    X_train=X_train.transpose(0,1,3,2)
    # X_train = proces(X_train,k) 
    # print(X_train.shape)
    # print(y_train.shape)
    return X_train, y_train      
    

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
            temp1 = math.sqrt(3)*X_vpm[j,i,:,:]*X_ipm[j,i,:,:]*np.cos(np.deg2rad(X_vpa[j,i,:,:]-X_ipa[j,i,:,:]))/1000000
            temp2 = math.sqrt(3)*X_vpm[j,i,:,:]*X_ipm[j,i,:,:]*np.sin(np.deg2rad(X_vpa[j,i,:,:]-X_ipa[j,i,:,:]))/1000000
            
            # print("temp1.shape",temp1.shape)
            st2.append(temp1)
            st4.append(temp2)
        st1.append(st2)
        st3.append(st4)
    st1 = np.array(st1)
    st3 = np.array(st3)
    # print(st1.shape)
    # print(st3.shape)
    return st1,st3
    
def diff(listA,listB):
    #求交集的两种方式
    retA = [i for i in listA if i in listB]          
    if retA:
        return True 
    else:
        return False    
# |Transformer_Planned
def rd_vpm(ii):
    path1 = "data/"
    df  = pd.read_csv(path1 +'S'+str(ii)+'.csv')
    dt = df.loc[df["word"].str.contains("Transformer_Trip|Transformer_Lightning|Transformer_Planned")].index
    print(dt.shape)
    dt2 = df.loc[df["word"].str.contains("Line_Trip|Line_Lightning")==False ].index
    # df[df["col1"].str.contains('this|that')==False
    print(dt2.shape)
    
    path2 = "../../../../../"
    dg = pd.read_csv(path2 +'muti.csv')
    print("dg.shape",dg.shape)
    dg = dg[(dg["v"]==4) | (dg["v"]==7)]
    ll = dg["new"].values.tolist()
    
    
    # print(ll)
    
    dt3 = df.loc[df.word2.isin(ll)].index
    print(dt3.shape)
    ind1 = list(set(dt)^set(dt2))
    ind2 = list(set(dt)^set(dt3))
    df.label[dt2] = 0   
    df.label[dt] = 1
    df.label[ind2] = 1
    
    
    # print(type()))
    # print(set(dt)^set(dt3))
    # print(set(dt)^set(dt3))
    dc1 = df.iloc[dt]
    # print(dc1["label"])
    dc2 = df.iloc[ind1]
    dc3 = df.iloc[ind2]
    # print(dcc["label"])
    df = pd.concat([dc1,dc2])
    df = pd.concat([df,dc3])
    y = df.pop("label").values
    		

    # df = df[["max_v_dn","mean_v_dn","min_v_dn","min_v_up","mean_v_up","max_v_up",
        # "max_p_dn","mean_p_dn","min_p_dn","min_p_up","mean_p_up","max_p_up"]]
    
    df.pop("word")
    df.pop("word2")
    df = df.values
    print(collections.Counter(y))
    print(df.shape)
    print(y.shape)
    return df,y
# def ff2():
    
def rd_rof(ii):
    path1 = "data/"
    df  = pd.read_csv(path1 +'S'+str(ii)+'.csv')
    full_ind = df.index
    path2 = "../../../../../"
    dg = pd.read_csv(path2 +'muti.csv')
    # print("dg.shape",dg.shape)
    dg = dg[(dg["v"]==7)]
    ll = dg["new"].values.tolist()
    print(df.shape[0])
    st = []
    
    dt2 = df.loc[df["word"].str.contains("Line_Trip|Line_Lightning")==True ].index
    dt4 = df.loc[df["word"].str.contains("Transformer")==True ].index
    dt3 = df.loc[df["word"].str.contains("Transformer_Trip|Transformer_Lightning|Transformer_Planned")].index
    dt1 = df.loc[df.word2.isin(ll)].index
    ind1 = bingji(dt2,dt4)
    ind1 = bingji(dt1,ind1)
    # print(len(dt2))

    # print("  "+str(len(dt1)))
    # print(len(ind1))
    #step1 get rid of the data
    ind3 = list(set(full_ind)^set(ind1))
    ind2 = bingji(dt3,ind3)
    # print(len(ind2))


    #step2 label data again 
    
    dg = pd.read_csv(path2 +'muti.csv')
    # print("dg.shape",dg.shape)
    dg = dg[(dg["v"]==4)]
    ll = dg["new"].values.tolist()
    dt1 = df.loc[df.word2.isin(ll)].index
    dt2 = df.loc[df["word"].str.contains("Trans")==True ].index
    df.label = 0
    df.label[dt2] = 1
    df.label[dt1] = 1
    res = df.iloc[ind2]
    res.pop("word")
    res.pop("word2")
    y = res.pop("label")
    
    return res.values,y.values
    
def jiaoji(listA,listB):
    return list(set(listA).intersection(set(listB)))

def bingji(listA,listB):
    return list(set(listA).union(set(listB)))
    
def get_split(ii):        
    X,y = rd_vpm(ii)
    # print(X.shape)
    # print(y.shape)
    spliter(ii,y)

def spliter(num,y3):        
    a = np.arange(0,y3.shape[0])
    tr,val = train_test_split(a,test_size=0.2)   
    print(tr.shape)
    print(val.shape)
    path2 = 'index2/'
    np.save(path2+'tr_'+str(num)+'.npy',tr) 
    np.save(path2+'val_'+str(num)+'.npy',val)


def rd_zeta(ii):
    # path1 = '../zeta/30/'
    path1 = '../zeta/150/'
    p1 = open(path1 +'X_S'+str(ii)+'.pickle',"rb")
    pk1 = pickle.load(p1)
    print(pk1.shape)
    st = []
    # 
    for i in range(pk1.shape[0]):
        temp = pk1[i,7]
        
        st.append(p_ind(np.argmax(temp)))
    st = np.array(st)
    print("rd_zeta shape",st.shape)
    return st
    
def p_ind(tt):
    # if(tt<180):
        # return 180
    if(tt>10500-180):
        return 10800-180
    else:
        return tt+205
        

def mid_data(k):

    path2 = '../lstm/index2/'
    tr = np.load(path2+'tr_'+str(k)+'.npy') 
    val = np.load(path2+'val_'+str(k)+'.npy') 
    list1 = tr.astype(int).tolist()
    list2 = val.astype(int).tolist()
    # list = ['rocof','vp_m','ip_m','f']
    X,y = rd_rof(k)
    X_train = X[list1]
    y_train = y[list1]
    X_val = X[list2]
    y_val = y[list2]    
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_val.shape)
    # print(y_val.shape)
    return X_train,y_train,X_val,y_val
    
 
    
def top():

    X_train,y_train,X_val,y_val = mid_data(1)
    for k in range(2,13+1):
        X_train2,y_train2,X_val2,y_val2 = mid_data(k)
        X_train = np.concatenate((X_train, X_train2), axis=0)
        y_train = np.concatenate((y_train, y_train2), axis=0)
        
        X_val = np.concatenate((X_val, X_val2), axis=0)
        y_val = np.concatenate((y_val, y_val2), axis=0)   

    
    print(collections.Counter(y_train))
    print(collections.Counter(y_val))
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    
    
    # X_train,y_train,X_val,y_val 
    # validation_set = (X_train,y_train,X_val,y_val )
    # pickle_out = open("cc.pickle","wb")
    # pickle.dump(validation_set, pickle_out, protocol=2)
    # pickle_out.close() 
    # print("save done")
    return X_train,y_train,X_val,y_val
    
def p_ind2(tt):
    if(tt<720):
        return 360
    if(tt>10800-360):
        return 10800-360
    else:
        return tt
        


    
def main():
    s1 = timeit.default_timer() 
    # for i in range(1,14):
        # get_split(i)
    # mid_data2(1)
    # rd_rof(3)
    top()
    # rd_vpm(1)
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
if __name__ == '__main__':  
    main()
