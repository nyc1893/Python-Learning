
# This is for Feq CWT^2 use
# import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import math
import pandas as pd
import pywt
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
import heapq
# from sklearn.model_selection import learning_curve, GridSearchCV  
# from sklearn.svm import SVC    
# from sklearn import tree
# from sklearn.ensemble import RandomForestClassifier 
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# import skopt
# from skopt import gp_minimize, forest_minimize
# from skopt.space import Real, Categorical, Integer
# from skopt.plots import plot_convergence
# from skopt.plots import plot_objective, plot_evaluations
# from skopt.utils import use_named_args
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




    




def rd2(k,i):
    path1 = '../pickleset2/'

    # list = ['ip_m','vp_a','va_m','va_a','vb_m','vb_a','vc_m','vc_a','rocof']
    list = ['rocof','vp_m','ip_m']

    p1 = open(path1 +'X_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")

    pk3  = pd.read_csv(path1 +'y_S'+str(k)+'_'+str(list[2])+'_6.csv')
    
    # pk3 = pk3[:,1]   
    # pk3 = pk3.astype(np.int32)      

    pk1 = pickle.load(p1)

    X_train = pk1
    y_train = pk3.values
    # X_train, y_train  = rm(X_train, y_train)


    return X_train, y_train    
    

def mid(ii,label):   
    X_train, y_train = rd2(ii,0)
    # print(type(np.where(y_train==3)[0]))
    a = np.where(y_train==label)[0]
    
    if(a.shape[0]>0):
        df = X_train[a]

        df =  np.squeeze(df)
        df = df.reshape(-1,10800)
        
        if(label ==3):
            print("Oscillation in dataset of ",ii)
        elif(label ==0):
            print("unplanned Line in dataset of ",ii)            
        elif(label ==1):
            print("unplanned Transformer in dataset of ",ii)    
        elif(label ==2):
            print("Freq in dataset of ",ii)    
        elif(label ==4):
            print("planned Line in dataset of ",ii)    

        elif(label ==5):
            print("planned Transformer in dataset of ",ii)    
            
        print("Rocof:")
        print("\t All Max:",np.max(np.max(df,axis=1)))
        print("\t All Min:",np.min(np.min(df,axis=1)))
        print("\t Mean Max:",np.mean(np.max(df,axis=1)))
        print("\t Mean Mmin:",np.mean(np.min(df,axis=1)))
        print("\t Mean mean:",np.mean(np.mean(df,axis=1)))
        
        X_train, y_train = rd2(ii,1)
        print("vp_m:")
        df = X_train[a]
        df =  np.squeeze(df)
        df = df.reshape(-1,10800)   
        print("\t All Max:",np.max(np.max(df,axis=1)))
        print("\t All Min:",np.min(np.min(df,axis=1)))    
        print("\t Mean Max:",np.mean(np.max(df,axis=1)))
        print("\t Mean Mmin:",np.mean(np.min(df,axis=1)))
        print("\t Mean mean:",np.mean(np.mean(df,axis=1)))    
        
        X_train, y_train = rd2(ii,2)
        print("ip_m:")
        df = X_train[a]
        df =  np.squeeze(df)
        df = df.reshape(-1,10800)   
        print("\t All Max:",np.max(np.max(df,axis=1)))
        print("\t All Min:",np.min(np.min(df,axis=1)))    
        print("\t Mean Max:",np.mean(np.max(df,axis=1)))
        print("\t Mean Mmin:",np.mean(np.min(df,axis=1)))
        print("\t Mean mean:",np.mean(np.mean(df,axis=1)))  


def save(label):  
    path = "gen/stat/"
    size = (13,5)
    m1 = np.zeros(size)
    m2 = np.zeros(size)
    m3 = np.zeros(size)
    for ii in range(1,14):
        X_train, y_train = rd2(ii,0)
        # print(type(np.where(y_train==3)[0]))
        a = np.where(y_train==label)[0]

        if(a.shape[0]>0):
            df = X_train[a]

            df =  np.squeeze(df)
            df = df.reshape(-1,10800)
            m1[ii-1,0] = np.max(np.max(df,axis=1))
            m1[ii-1,1] = np.min(np.min(df,axis=1))
            m1[ii-1,2] = np.mean(np.max(df,axis=1))
            m1[ii-1,3] = np.mean(np.min(df,axis=1))
            m1[ii-1,4] = np.mean(np.mean(df,axis=1))

            X_train, y_train = rd2(ii,1)
            df = X_train[a]
            df =  np.squeeze(df)
            df = df.reshape(-1,10800)   
            m2[ii-1,0] = np.max(np.max(df,axis=1))
            m2[ii-1,1] = np.min(np.min(df,axis=1))
            m2[ii-1,2] = np.mean(np.max(df,axis=1))
            m2[ii-1,3] = np.mean(np.min(df,axis=1))
            m2[ii-1,4] = np.mean(np.mean(df,axis=1))

            X_train, y_train = rd2(ii,2)
            df = X_train[a]
            df =  np.squeeze(df)
            df = df.reshape(-1,10800)   
            m3[ii-1,0] = np.max(np.max(df,axis=1))
            m3[ii-1,1] = np.min(np.min(df,axis=1))
            m3[ii-1,2] = np.mean(np.max(df,axis=1))
            m3[ii-1,3] = np.mean(np.min(df,axis=1))
            m3[ii-1,4] = np.mean(np.mean(df,axis=1))
        
    m1 = pd.DataFrame(m1)
    m2 = pd.DataFrame(m2)
    m3 = pd.DataFrame(m3)
    
    m1.columns = ['max','min','maxBar','minBar','meanBar']
    m2.columns = ['max','min','maxBar','minBar','meanBar']
    m3.columns = ['max','min','maxBar','minBar','meanBar']

    m1.to_csv(path+"rocof_"+str(label)+".csv",index =None)
    m2.to_csv(path+"vp_m_"+str(label)+".csv",index =None)
    m3.to_csv(path+"ip_m_"+str(label)+".csv",index =None)    
    
def load_data(ii):   
    pickle_in = open("gen/X_"+str(ii)+".pickle","rb")
    X_train,y_train,X_val,y_val= pickle.load(pickle_in)
  
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    return X_train,y_train,X_val,y_val


def select(X):

    res =[]
    for j in range(0,X.shape[0]):
        nums = []
        for i in range(0,23):
            temp = X[j,i,:,0]
            nums.append(np.max(temp)-np.min(temp))
            
        max_num_index_list = map(nums.index, heapq.nlargest(3, nums))
        ll = list(max_num_index_list)
        temp = X[j,ll,:,:]
        res.append(temp)

    return np.array(res)
    

def main(cc):
    s1 = timeit.default_timer()  
    # for k in range(1,14):
        # mid(k,5)
    save(cc)
    
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
if __name__ == '__main__':  
    cc = int(sys.argv[1])
    main(cc)
