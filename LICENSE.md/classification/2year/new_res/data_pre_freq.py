
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

from scipy.stats import pearsonr
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


def rd_zeta(ii):
    # path1 = '../zeta/30/'
    path1 = '../zeta_all/150/'
    p1 = open(path1 +'X_S'+str(ii)+'.pickle',"rb")
    pk1 = pickle.load(p1)
    print(pk1.shape)
    st = []
    # 
    for i in range(pk1.shape[0]):
        temp = pk1[i,0]
        
        st.append(p_ind(np.argmax(temp)))
    st = np.array(st)
    # print(ll)
    return st

def rd_rof(ii,feature):
    path1 = '../../../pickleset2/'
    list = ['rocof','vp_m','ip_m','f']
    # feature = 1
    p1 = open(path1 +'X_S'+str(ii)+'_'+str(list[feature])+'_6.pickle',"rb")

    pk3  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[feature])+'_6.csv')
   
    pk1 = pickle.load(p1)

    X_train = pk1
    y_train = pk3.values
    y_train = y_train[:,0]

    # X_train, y_train  = rm3(X_train, y_train)
    ind1 = np.where(y_train!=2)[0]
    ind2 = np.where(y_train==2)[0]
    y_train[ind1] = 0
    y_train[ind2] = 1
    
    path2 = "../rm_index/"
    tr = np.load(path2+'S'+str(ii)+'.npy') 
    # X_train = X_train[tr]
    # y_train = y_train[tr]    


    print(X_train.shape)
    print(y_train.shape)    
    return X_train, y_train    
    
def choose(X):

    nums = []
    for i in range(0,23):
        temp = X[i,:,0]
        nums.append(np.max(temp)-np.min(temp))
    # nums.sort()
    # print(nums)
    max_num_index_list = map(nums.index, heapq.nlargest(1, nums))

    ll = list(max_num_index_list)
    print("Largest dif ref:",ll[0])
    return np.squeeze(X[ll[0],:,0]),ll[0]
    
def pearsonrSim(x,y):
    '''
    皮尔森相似度
    '''
    return pearsonr(x,y)[0]
    

def cal_area(mm,temp):
    sup =0
    sdown =0
    for i in range(temp.shape[0]):
        if(temp[i]>=mm):
            sup+=temp[i]-mm
        else:
            sdown+= -(temp[i]+mm)
    return sup,sdown
    
def save_freq(ii):

    X_train, y_train  = rd_rof(ii,0)
    indd =rd_zeta(ii)
    st =[]
    for j in range(0,X_train.shape[0]):
        list_up =[]
        list_down = []
        list_r =[]
        list_a1 =[]
        list_a2 =[]
        list_sim =[]
        # list_up
        flag =0
        kk = np.zeros(9)
        # y = df1["label"].values[j]
        y = y_train[j]
        ref,ll  = choose(X_train[j,:,:,:])
        for i in range(0,23):
            x = np.squeeze(X_train[j,i,:,:])
            mm = np.mean(x)
            temp = x[indd[j]-180:indd[j]+180]
            up = np.max(temp)-mm
            down = mm-np.min(temp)
            rate = down/up
            sup,sdown = cal_area(mm,temp)
            a1 = sdown/sup
            a2 = sdown/(sup+sdown)
            list_up.append(up)
            list_down.append(down)
            list_r.append(rate)
            list_a1.append(a1)
            list_a2.append(a2)
            list_sim.append(pearsonrSim(ref,x))
            
        kk[0] = np.mean(list_up)
        kk[2] = np.std(list_up)
        kk[1] = np.mean(list_down)
        kk[3] = np.std(list_down)

        kk[4] = np.mean(list_r)
        kk[5] = np.mean(list_a1)
        kk[6] = np.mean(list_a2)
        kk[7] = y
        kk[8] = np.min(list_sim)
        st.append(kk)
        
    df = pd.DataFrame(st)
    df.columns = ["upBar","downBar","upstd","downstd","rate","a1","a2","label","sim"]
    df.to_csv("ml/full"+str(ii)+".csv",index =None)
    print("save done")
    print(df.tail())
    print(df.shape)         

def diff(listA,listB):
    #求交集的两种方式
    retA = [i for i in listA if i in listB]          
    if retA:
        return True 
    else:
        return False    

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



    
def mid_data(k):

    path2 = 'index5/'
    tr = np.load(path2+'tr_'+str(k)+'.npy') 
    val = np.load(path2+'val_'+str(k)+'.npy') 
    list1 = tr.astype(int).tolist()
    list2 = val.astype(int).tolist()
    # list = ['rocof','vp_m','ip_m','f']
    X,y = rd_rof(k,1)
   
    X_train = X[list1]
    y_train = y[list1]
    X_val = X[list2]
    y_val = y[list2]
    return X_train,y_train,X_val,y_val
    
 
    
def funn(ii):
    df = pd.read_csv("ml/full"+str(ii)+".csv")
    dt = pd.read_csv("../py2/data/S"+str(ii)+".csv")
    df["word"] = dt["word"]
    df["word2"] = dt["word2"]
    
    print(df.head())
    df.to_csv("ml/full"+str(ii)+".csv",index = None)
    
def p_ind2(tt):
    if(tt<720):
        return 360
    if(tt>10800-360):
        return 10800-360
    else:
        return tt
        
def p_ind(tt):
    # if(tt<180):
        # return 180
    if(tt>10500-180):
        return 10800-180
    else:
        return tt+205
        

    
def main():
    s1 = timeit.default_timer() 
    for i in range(1,13+1):
        funn(i)

    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
if __name__ == '__main__':  
    main()
