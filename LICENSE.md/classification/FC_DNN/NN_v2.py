#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
creating the dataset
"""



# import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras import optimizers,regularizers

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import timeit
import pickle

start = timeit.default_timer()

def removePlanned(X,y):
    """
    THIS FUNCTION REMOVES THE PLANNED EVENTS FROM THE EVENT DATASET
    """
    
    X_new=[]
    y_new=[]
    for i in range(len(y)):
        #print(i)
    
        if y[i]==0:
            y_new.append(0)
            X_new.append(X[i,:,:,:])
    
        elif y[i]==1:
            y_new.append(1)
            X_new.append(X[i,:,:,:])
    
            
        elif y[i]==2:
            y_new.append(2)
            X_new.append(X[i,:,:,:])
        
        elif y[i]==3:
            y_new.append(3)
            X_new.append(X[i,:,:,:])
        

    return  np.array(X_new), np.array(y_new)


def separatePMUs(X,y):
    
    """
    This function separates features and their corresponding labels for each PMU
    to make more events 
    """
    
    num_case=X.shape[0]
    num_pmu=X.shape[1]
    num_sample=X.shape[2]
    X=X.reshape(num_case*num_pmu,num_sample)
    y2=[]
    for i in range(len(y)):
        if y[i]==0:
            for j in range(num_pmu):
                y2.append(0)
                
        elif y[i]==1:
            for j in range(num_pmu):
                y2.append(1)
                
        elif y[i]==2:
            for j in range(num_pmu):
                y2.append(2)
                
        elif y[i]==3:
            for j in range(num_pmu):
                y2.append(3)
        elif y[i]==4:
            for j in range(num_pmu):
                y2.append(4)       
        elif y[i]==5:
            for j in range(num_pmu):
                y2.append(5)    
    return X,np.array(y2)


'''
num_filter=[2,5,10,20,50]
h_filter=[2,5,10,20]
w_filter=[2, 10,100, 1000]
'''
def fun(a,b):
    if a == b:
        return 1
    else:
        return 0
        
def check_X(X,y):
    j = 0 
    num = X.shape[0]
    list= []
    for i in range(0,num-1):
        d1 = X[i,0,:,:]
        # d11 = y[i] 
        d2 = X[i+1,0,:,:]
        # d22 = y[i+1] 
        c1 = X[i,7,:,:]
        c2 = X[i+1,7,:,:]
        
        
        # and  (c1 ==c2).all()
        if (d1 ==d2).all() and (c1 ==c2).all() and ((y[i] == 0 and y[i+1] ==1) or (y[i] == 1 and y[i+1] ==0) or  (y[i] == 1 and y[i+1] ==1)):
            list.append(i)
            list.append(i+1)

            j = j+1
    print('total:',j)
    return list
            
def get_label (X,y):
    # X,y = read_data3()
    ll = check_X(X,y)
    # print(ll)
    
    df = pd.DataFrame(y)
    df.columns = ['v']
    
    cc = np.zeros(y.shape[0], dtype=np.int)
    cc[ll] = 1
    df['LL'] = cc
    # print(df['LL'].value_counts())
    df['L1'] = df.apply(lambda x: fun(x.v,0),axis =1)
    
    # print(df['L1'].value_counts())
    # df['L1'] = df.apply(lambda x: fun(x.LL,1),axis =1)
    df.loc[ll,'L1'] = 1 
    # print(df['L1'].value_counts())
    
    df['L2'] = df.apply(lambda x: fun(x.v,1),axis =1)
    # print(df['L2'].value_counts())
    df.loc[ll,'L2'] = 1 
    # print(df['L2'].value_counts())
    
    df['L3'] = df.apply(lambda x: fun(x.v,2),axis =1)
    df['L4'] = df.apply(lambda x: fun(x.v,3),axis =1)
    df.pop('v')
    df.pop('LL')

    return df


path = 'pickleset/'
num = 6

for i in range(1,num+1):
    p1 = open(path +"X_S"+str(i)+"_rocof_6.pickle","rb")
    pk1 = pickle.load(p1)
    
    fps=60
    start_crop=int(fps*60*4)
    stop_crop=int(fps*60*8)
    pk1 = pk1[:,:,start_crop:stop_crop,:]

    p2 = open(path + "y_S"+str(i)+"_rocof_6.pickle","rb")
    pk2 = pickle.load(p2)
    
    pk1,pk2 =removePlanned(pk1,pk2)
    
    yn = get_label (pk1,pk2)

    locals()['X_train'+str(i)], locals()['X_val'+str(i)], locals()['y_train'+str(i)], locals()['y_val'+str(i)]=train_test_split(pk1, yn, test_size=0.25)
print(X_train1.shape)
print(X_val1.shape)
print(y_train1.shape)
print(y_val1.shape)

for i in range(1,num+1):
    X_train1=np.concatenate((X_train1, locals()['X_train'+str(i)]), axis=0)
    X_val1=np.concatenate((X_val1, locals()['X_val'+str(i)]), axis=0)
    y_train1=np.concatenate((y_train1, locals()['y_train'+str(i)]), axis=0)
    y_val1=np.concatenate((y_val1, locals()['y_val'+str(i)]), axis=0)    
    
    
print(X_train1.shape)
print(X_val1.shape)
print(y_train1.shape)
print(y_val1.shape)
path2 = '2016-'
sav = 1
if sav == 1:
    training_set=(X_train1, y_train1)
    pickle_out = open(path2+"tr_set.pickle","wb")
    pickle.dump(training_set, pickle_out, protocol=2)
    pickle_out.close() 


    validation_set = (X_val1, y_val1)
    pickle_out = open(path2+"va_set.pickle","wb")
    pickle.dump(validation_set, pickle_out, protocol=2)
    pickle_out.close() 
