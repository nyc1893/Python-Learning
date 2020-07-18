


import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import math
import pandas as pd
from datetime import datetime, timedelta
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import InputLayer, Input
# from tensorflow.keras.layers import Reshape, MaxPooling2D
# from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
# from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import load_model
# from tensorflow.keras import optimizers,regularizers

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import sys
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

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


    
# /home/aghasemkhani/DOE_Recovered/8weeks_2/

def read_data3():  
    path = './'
    path3 = '../pickleset/'
    k = 3
    p1 = open(path3+ "X_S"+str(k)+"_rocof_6.pickle","rb")
    X_train = pickle.load(p1)
    

    p1 = open(path3+ "y_S"+str(k)+"_rocof_6.pickle","rb")
    y_train = pickle.load(p1)
    
    
    
    fps=60
    start_crop=int(fps*60*4)
    stop_crop=int(fps*60*8)
    X_train = X_train[:,:,start_crop:stop_crop,:]
    # X_train=np.concatenate((X_train, X_train2), axis=0)
    # X_test=X_test[:,:,start_crop:stop_crop,:]
    # y_train=np.concatenate((y_train, y_train2), axis=0)
    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)
    
    # df = pd.Series(y_train)
    # print(df.value_counts())
    
    
    # only shuffle X_train part 
    X_train, y_train =removePlanned(X_train, y_train)
    # X_train, y_train=separatePMUs(X_train, y_train)
    
    
    
    # X_train, y_train = shuffle(X_train, y_train)   

    # X_test, y_test=removePlanned(X_test, y_test)
    # X_test, y_test= separatePMUs(X_test,y_test)
 
    
    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)

    
    return X_train,y_train   
    


    
    
def check_y_test(y_test):
    # X_test,y_test = read_data()
    i = 0
    num = y_test.shape[0]/23
    # num = 2
    # j = 0
    flag = 0
    for i in range(1,int(num)+1):
        ind1 = 23*(i-1)
        ind2 = 23*i
        
        data =  y_test[ind1:ind2]
        # print(data)
        for j in range(1,23):
            if(data[0] != data[j]):
                flag = 1
        if flag == 1:
            print('Not equal in ', i)   
    return flag
  
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
    
    
def find_duplicate():
    path = '../../../'
    df = pd.read_csv(path + 'B_Training.csv')
    # start_time= df['Start']
    # datetime_start = datetime.strptime(start_time, '%m/%d/%Y %H:%M') 
    # df= df.sort_values(by=['End','Event'])
    
    # sorted_df10 = df['2016-2-26':'2016-4-7']
    
    df = df.values
    # num = df.shape[0]
    # print(num)


    j = 0
    j2= 0
    list1 = []
    list2 = []
    for i in range(283,283+336 -2):
        d1 = df[i,1]
        d2 = df[i+1,1]

        a1 = df[i,2]
        a2 = df[i+1,2]
        
        c1 = df[i,4]
        c2 = df[i+1,4] 
        
        b1 = df[i,3]
        b2 = df[i+1,3]
        
        
        if str(b1).find("Planned") == 1:
            c= 0
        else:
            if d1 == d2 and b1 == b2 and c1 ==c2 and a1 ==a2:
                j = j+1
                list1.append(d1)
                
            elif d1 == d2 and b1 != b2 :
                j2 = j2+1
                list2.append(d1)
        
    print(j)
    print(list1)
    print(j2)
    print(list2)
    # print(df[283-1,1])
    # print(df[283,1])
    # print(df[283+336-2,1])
    # print(df[283+336-1,1])
    # print(df[283+336-1,2])
    # print(df[283+336-1,3])
    # print(df[283+336-1,4])
    # print(df[283+336,1])
    print(d1)
    print(b1)
    
def fun(a,b):
    if a == b:
        return 1
    else:
        return 0
        
def get_label ():
    X,y = read_data3()
    ll = check_X(X,y)
    print(ll)
    
    df = pd.DataFrame(y)
    df.columns = ['v']
    
    cc = np.zeros(y.shape[0], dtype=np.int)
    cc[ll] = 1
    df['LL'] = cc
    # print(df['LL'].value_counts())
    df['L1'] = df.apply(lambda x: fun(x.v,0),axis =1)
    
    print(df['L1'].value_counts())
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
    # print(df['L1'].value_counts())
    # print(df['L2'].value_counts())
    # print(df['L3'].value_counts())
    # print(df['L4'].value_counts())
    return df
        
def main():
    
    df = get_label()
    print(df.head())

if __name__ == '__main__':  
    # global best_accuracy 
    main()   
    
