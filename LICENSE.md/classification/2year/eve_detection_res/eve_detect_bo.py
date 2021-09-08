# This one is to get the event detection accruacy result based on the data generated in cal2.py
# Can be fond in py_file/code/cnn2/ml2/eve_det/
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
import statsmodels.api as sm 

import sys
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

   
def classit2(data,t1,t3,t5):

    # list5 =[0,2,4,6,7]
    
    temp = data[0,:]
    for i in range(temp.shape[0]):
        if(temp[i]>t1):
            # vpm
            return 1

            
    temp = data[2,:]
    for i in range(temp.shape[0]):
        if(temp[i]>t3):
            # vpm
            return 1
      
            
    temp = data[4,:]
    for i in range(temp.shape[0]):
        if(temp[i]>t5):
            # vpm
            return 1
            


    return 0
    
    
def get_nor():
    ii =1
    path1 ="../data_pre/nor_ind_"
    df1 = np.load(path1+str(ii)+'.npy')
    
    for ii in range(2,8):
        df2 = np.load(path1+str(ii)+'.npy')
        df1 = np.concatenate((df1, df2), axis=0)
    print(df1.shape)
    return df1
    
def get_eve():
    ii =1
    path1 ="../data_pre/eve_ind_"
    df1 = np.load(path1+str(ii)+'.npy')
  
    for ii in range(2,14):
        df2 = np.load(path1+str(ii)+'.npy')
        df1 = np.concatenate((df1, df2), axis=0)
    print(df1.shape)
    return df1
    
 
def fun(t1):
 
    pred =[]
    real = []
    for i in range(dt1.shape[0]):
        temp = dt1[i]
        real.append(0)
        pred.append(classit(temp,t1))
    
    for i in range(dt2.shape[0]):
        temp = dt2[i]
        real.append(1)
        pred.append(classit(temp,t1))    
    
    
    f1_macro = f1_score(pred,real,average='macro')
    print(f1_macro)
    return f1_macro
    
def fun3(t1,t3,t5,t7,t8):
 
    pred =[]
    real = []
    for i in range(dt1.shape[0]):
        temp = dt1[i]
        real.append(0)
        pred.append(classit2(temp,t1,t3,t5,t7,t8))
    
    for i in range(dt2.shape[0]):
        temp = dt2[i]
        real.append(1)
        pred.append(classit2(temp,t1,t3,t5,t7,t8))    
    
    
    acc = accuracy_score(pred,real)
    print(acc)
    return acc    
    
def get_index():
    path2 ="index2/"
    arr = np.arange(0,dt1.shape[0])
    np.random.shuffle(arr)
    np.save(path2+'bo.npy',arr) 
    
def fun2(t1,t3,t5):
    path2 ="index2/"
    arr = np.load(path2+'bo.npy')
    j = 4
    num = round(0.2*arr.shape[0])
    test_index = list(arr[j*num:num*(j+1)])

    train_index=list(set(arr)-set(test_index))
        
    pred =[]
    real = []
    
    pred2 =[]
    real2 = []    
    
    for i in train_index:
        temp = dt1[i]
        real.append(0)
        pred.append(classit2(temp,t1,t3,t5))
    
    for i in train_index:
        temp = dt2[i]
        real.append(1)
        pred.append(classit2(temp,t1,t3,t5))    
    
    # for i in test_index:
        # temp = dt1[i]
        # real2.append(0)
        # pred2.append(classit2(temp,t1,t3,t5,t7,t8))
    
    # for i in test_index:
        # temp = dt2[i]
        # real2.append(1)
        # pred2.append(classit2(temp,t1,t3,t5,t7,t8))    
        
    acc = accuracy_score(pred,real)
    # acc2 = accuracy_score(pred,real)
    print(acc)
    return acc    
    
from bayes_opt import BayesianOptimization    

# def black_box_function(q1,d1,L):
    # qq1 = int(round(q1))
    # qq2 = qq1
    # dd1 = int(round(d1))
    # dd2 = dd1
    # L2 = int(round(L))
    # turb = 'mit'

    # return fun(t1,t2,t3,t4)
    
    
def main():
    s1 = timeit.default_timer()  
    global dt1,dt2
    dt1 = get_nor()
    dt2 = get_eve()
    get_index()
    # Bounded region of parameter space
    pbounds = {'t1': (0.01, 10)

                }
                
    # Bounded region of parameter space
    pbounds2 = {'t1': (0.001, 10), 
    # 't2': (0.01, 3),
               't3': (0.001, 5),
               # 't4': (0.05, 5),
                't5': (0.001, 5), 
                # 't6': (0.01, 5),
                # 't7': (0.01, 0.2),
                # 't8': (0.01, 0.2)              
                }
    optimizer = BayesianOptimization(
        f=fun2,
        pbounds=pbounds2,
        random_state=int(sys.argv[1]),
    )
    optimizer.maximize(
        init_points=20,
        n_iter=180
    )
    tt = str(optimizer.max)
    with open('30.txt','a+') as f:    #设置文件对象
        f.write(tt+'\n')                 #将字符串写入文件中
    print(tt)        
    s2 = timeit.default_timer()  
    print ('Runing time is Hour:',round((s2 -s1)/3600,2))
        

if __name__ == '__main__':  



    
    
    main()
    # get_nor()
