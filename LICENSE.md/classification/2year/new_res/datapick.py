
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
start = timeit.default_timer()

def remove_z(str):
    if str[0] == '0':
        return str[1]
    else:
        return str
        


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
    

def rd_rof(ii):
    path1 = "ml/"
    df  = pd.read_csv(path1 +'full'+str(ii)+'.csv')
    full_ind = df.index
    path2 = "../../../../../"
    dg = pd.read_csv(path2 +'muti.csv')
    # print("dg.shape",dg.shape)
    dg = dg[(dg["v"]==7)]
    ll = dg["new"].values.tolist()
    # print(df.shape[0])
    st = []
    
    df2 = df.replace([np.inf, -np.inf], np.nan)
    df2 = df2.dropna()    
    ind_null = df2.index
    
    
    
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
    dg = dg[(dg["v"]==5)]
    ll = dg["new"].values.tolist()
    dt1 = df.loc[df.word2.isin(ll)].index
    dt2 = df.loc[df["word"].str.contains("Freq")==True ].index
    df.label = 0
    df.label[dt2] = 1
    df.label[dt1] = 1
    res = df.iloc[ind2]

    
    


    res.pop("word")
    res.pop("word2")
    y = res.pop("label")
    
    res = res.replace([np.inf, -np.inf], np.nan)
    ind = res[res.isnull().T.any()].index
    res.loc[ind,:] = -1
    # print(res.loc[ind])
    
    
    return res.values,y.values
    
def jiaoji(listA,listB):
    return list(set(listA).intersection(set(listB)))

def bingji(listA,listB):
    return list(set(listA).union(set(listB)))
    
def get_split(ii):        
    X,y = rd_rof(ii)
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

    path2 = 'index2/'
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

    
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
 
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
    # mid_data(1)
    # rd_rof(3)
    top()
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
if __name__ == '__main__':  
    main()
