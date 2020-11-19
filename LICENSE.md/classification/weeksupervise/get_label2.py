

# import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import math
import pandas as pd
import time  
from sklearn import metrics  
import pickle as pickle  
import pandas as pd


import pickle

import datetime


from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time  
from sklearn import metrics  
import pickle as pickle  


import timeit
import time  
from sklearn import metrics  
import pickle as pickle  
import pandas as pd
import random

from rno_fun import data_lowfreq

    
    


  

      
def run():
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  

    train_x, train_y,test_x, test_y =readdata()   

   

    df = pd.DataFrame(df,columns=list)
    df2 = pd.DataFrame(df2,columns=list)
    
    df.to_csv('Ltest_34.csv',index =None)
    df2.to_csv('Ltrain_12.csv',index =None)
    

    print('save done!')

    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
    
def rd1(i):
    path = 'data/'
    X_train =  pd.read_csv(path +'D'+str(i)+'_Ltr.csv')
    X_train = X_train.rename(columns={'21':'label'})
    y_train = X_train.pop('label')
    y_train = y_train.astype('int') 
    # print(X_train.head())
    
    X_test =  pd.read_csv(path +'D'+str(i)+'_Ltest.csv')
    X_test = X_test.rename(columns={'21':'label'})
    y_test = X_test.pop('label')
    y_test = y_test.astype('int') 
    return  X_train.values,y_train.values,X_test.values,y_test.values
    
def readdata():
    X_train,y_train, X_val, y_val= rd1(1)
    for i in range(2,7):
        X_train2,y_train2, X_val2, y_val2= rd1(i)
        X_train = np.concatenate((X_train,X_train2), axis=0)
        y_train = np.concatenate((y_train,y_train2), axis=0)
        X_val = np.concatenate((X_val,X_val2), axis=0)
        y_val = np.concatenate((y_val,y_val2), axis=0)
    # X_train = np.concatenate((X_train,X_val), axis=0)
    # y_train = np.concatenate((y_train,y_val), axis=0)
    # print('unique-ytran',np.unique(y_train))
    # print('X_train.shape',X_train.shape) 
    # print('y_train.shape',y_train.shape) 
  
    return X_train,y_train,X_val,y_val
def gen_noise():
    X_train,y_train,X_val,y_val = readdata()
    print(y_val.shape[0])

    v = round(y_val.shape[0]*0.2)
    print(v)
    for j in range(1,8+1):
        globals()['y'+str(j)] = y_val
        globals()['y'+str(j)] = globals()['y'+str(j)][:,np.newaxis]
        list = []
        for i in range(0,v):
            list.append(random.randint(0,y_val.shape[0]-1))
        # print(list)
        for i in list:
            globals()['y'+str(j)][i] = random.randint(0,5)
    y0 = y_val
    y0 = y0[:,np.newaxis]
    y = y0
    for i in range(1,8+1):
        y  = np.concatenate((y,globals()['y'+str(i)]), axis=1)
    print(y.shape)

def main():
    gen_noise()
    
if __name__ == '__main__':  

    main()

