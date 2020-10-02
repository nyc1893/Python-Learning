
# This is to get va_a,va_m data
#  Already deal with the mutilabel
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import pandas as pd
import pywt
from sklearn.decomposition import PCA

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Reshape, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers,regularizers
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
# from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection  import StratifiedKFold
from sklearn.model_selection  import cross_val_score

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
            
        elif y[i]==6:
            y_new.append(6)
            X_new.append(X[i,:,:,:])
        
        elif y[i]==7:
            y_new.append(7)
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
        elif y[i]==6:
            for j in range(num_pmu):
                y2.append(6)       
        elif y[i]==7:
            for j in range(num_pmu):
                y2.append(7)    
    return X,np.array(y2)




def rd2(k,i):
    path1 = '../pickleset/'

    list = ['vp_m','vp_a','va_m','va_a']
    

    p1 = open(path1 +'X2_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")
    p3  = open(path1 +'y2_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")    
    pk3 = pickle.load(p3)
    print(pk3.shape)
    pk3 = deal_label(pk3)
    print(pk3.shape)
    # pk3 = pk3[:,1]   
    # pk3 = pk3.astype(np.int32)          
    pk1 = pickle.load(p1)

    fps=60

    start_crop=int(fps*60*4)
    stop_crop=int(fps*60*8)

    pk1=pk1[:,:,start_crop:stop_crop,:]
        

    # print(pk3[0:1])
    # print(pk3.shape)
    path2 = 'index2/'
    tr=np.load(path2 +'tr_' +str(k)+'.npy')
    val=np.load(path2 +'val_' +str(k)+'.npy')
    tr=tr.tolist()  
    val=val.tolist() 
    # b = a[c]
    
    pk1,pk3=removePlanned(pk1,pk3)

    X_train = pk1[tr]
    y_train = pk3[tr]
    X_val = pk1[val]
    y_val = pk3[val]
    
    print(X_train.shape) 
    print(X_val.shape) 
    print(y_train.shape) 
    print(y_val.shape) 
    """
    """
    return X_train, X_val, y_train, y_val    

    
def DWT(X_train):


    (ca1, cd1) = pywt.dwt(X_train, 'db2')
    (ca2, cd2) = pywt.dwt(ca1, 'db2')
    (ca3, cd3) = pywt.dwt(ca2, 'db2')
    (ca4, cd4) = pywt.dwt(ca3, 'db2')
    (ca5, cd5) = pywt.dwt(ca4, 'db2')
    (ca6, cd6) = pywt.dwt(ca5, 'db2')
    
    num = 3
    pca_2 = PCA(n_components=num)
    
    ca6 = pca_2.fit_transform(ca6)
    ca5 = pca_2.fit_transform(ca5)    
    
    ca4 = pca_2.fit_transform(ca4)
    ca3 = pca_2.fit_transform(ca3)
    ca2 = pca_2.fit_transform(ca2)
    ca1 = pca_2.fit_transform(ca1)
    
    cd6 = pca_2.fit_transform(cd6)
    cd5 = pca_2.fit_transform(cd5)    
    cd4 = pca_2.fit_transform(cd4)
    cd3 = pca_2.fit_transform(cd3)
    cd2 = pca_2.fit_transform(cd2)
    cd1 = pca_2.fit_transform(cd1)    
    # print('cd1.shape',cd1.shape) 
    # print('cd2.shape',cd2.shape) 
    # print('cd3.shape',cd3.shape) 
    # print('cd4.shape',cd4.shape) 
    # print('cd5.shape',cd5.shape) 
    # print('cd6.shape',cd6.shape) 
    
    # print('ca1.shape',ca1.shape) 
    # print('ca2.shape',ca2.shape) 
    # print('ca3.shape',ca3.shape) 
    # print('ca4.shape',ca4.shape)    
    # print('ca5.shape',ca5.shape) 
    # print('ca6.shape',ca6.shape)     
    return ca6,ca5, ca4,ca3,ca2,ca1,cd6,cd5,cd4,cd3,cd2,cd1
    
def deal_label(y_test):
    path1 = 'pca_plot2/'
    y_test = pd.DataFrame(y_test)
    # print(y_test.head())
    y_test.columns = ['event',	'label']
    df = y_test.pop('event')
    # print(df.head())
    df = df.str.split('_')
    df2 = pd.DataFrame(df)
    y_test['year'] = df2['event'].str[0]
    y_test['month'] = df2['event'].str[1]
    y_test['day'] = df2['event'].str[2].astype(int)
    y_test['no'] = df2['event'].str[3]   

    y_test['new'] = y_test['year'].astype(str).str.cat(y_test['month'].astype(str),sep = '_')
    y_test['new'] = y_test['new'].str.cat(y_test['day'].astype(str),sep = '_')
    y_test['new'] = y_test['new'].str.cat(y_test['no'].astype(str),sep = '_')    

    y_test = y_test[['new','label']]
    
    

    df2 = pd.read_csv(path1+'trans2.csv')
    list = df2['0'].tolist()
    y_test['label'] = y_test['label'].astype("int")
    ind1 = y_test['new'].isin(list).tolist()
    # print(ind1)
    df2 = pd.read_csv(path1+'feq2.csv')
    list = df2['0'].tolist()
    ind2 = y_test['new'].isin(list).tolist()
    
    y_test.loc[ind1, 'label'] = 6
    y_test.loc[ind2, 'label'] = 7
    # y_test['label'].iloc[ind2] = 7
    
    
    y_test.pop('new')
    
    # print(y_test.loc[40:55])
    # y_test.to_csv('kankan.csv',index =None)
    return y_test.values
    
def dproc1():
    X_train, X_val, y_train, y_val = rd2(1,2)
    X_train,  y_train = separatePMUs (X_train,  y_train)
    X_val,  y_val = separatePMUs (X_val,  y_val)

    num= X_train.shape[0]

    p1 = np.concatenate((X_train,X_val))
    p1 = p1.reshape(p1.shape[0],-1)
    d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11 = DWT(p1)
    
    p3 = locals()['d'+str(0)] 
    X_train = p3[:num]
    X_val = p3[num:]    
        
    for i in range(1,5):
        p3 = locals()['d'+str(i)] 
        X_train2 = p3[:num]
        X_val2 = p3[num:]

        X_train = np.concatenate((X_train,X_train2), axis=1)
        X_val = np.concatenate((X_val,X_val2), axis=1)

    
    print('X_train.shape',X_train.shape) 
    print('y_train.shape',y_train.shape) 
    print('X_val.shape',X_val.shape) 
    print('y_val.shape',y_val.shape) 
    return  X_train,y_train, X_val, y_val
    

def dproc2():
    X_train, X_val, y_train, y_val = rd2(1,3)
    X_train,  y_train = separatePMUs (X_train,  y_train)
    X_val,  y_val = separatePMUs (X_val,  y_val)

    num= X_train.shape[0]

    p1 = np.concatenate((X_train,X_val))
    p1 = p1.reshape(p1.shape[0],-1)
    d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11 = DWT(p1)
    
    p3 = locals()['d'+str(0)] 
    X_train = p3[:num]
    X_val = p3[num:]    
        
    for i in range(1,5):
        p3 = locals()['d'+str(i)] 
        X_train2 = p3[:num]
        X_val2 = p3[num:]

        X_train = np.concatenate((X_train,X_train2), axis=1)
        X_val = np.concatenate((X_val,X_val2), axis=1)

    
    print('X_train.shape',X_train.shape) 
    print('y_train.shape',y_train.shape) 
    print('X_val.shape',X_val.shape) 
    print('y_val.shape',y_val.shape) 
    return  X_train,y_train, X_val, y_val

def data_pack():
    X_train,y_train, X_val, y_val= dproc1()
    X_train2,y_train, X_val2, y_val= dproc2()
    print('X_train.shape',X_train.shape) 
    print('X_train2.shape',X_train2.shape)     
    X_train = np.concatenate((X_train,X_train2), axis=1)
    X_val = np.concatenate((X_val,X_val2), axis=1)

    print(np.unique(y_train))
    print('X_train.shape',X_train.shape) 
    print('y_train.shape',y_train.shape) 
    print('X_val.shape',X_val.shape) 
    print('y_val.shape',y_val.shape) 
    # return  X_train,y_train, X_val, y_val
    
    
if __name__ == '__main__':  

    # rd2(1,3)
    data_pack()
