
#For osc detection use
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

def classit1(data,vec):
    for i in range(data.shape[0]):
        if(data[i]>vec[i]):
            return 1
    return 0
   
def classit2(data,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12):

    j = 5
    temp = data[0+j]
    # for i in range(data.shape[0]):
    if(temp>t1):
        return 1           
    temp = data[6+j]
    # for i in range(data.shape[0]):
    if(temp>t2):
        return 1
            
    temp = data[12+j]
    # for i in range(data.shape[0]):
    if(temp>t3):
        return 1      
    temp = data[18+j]
    # for i in range(data.shape[0]):
    if(temp>t4):
        return 1   
            
    temp = data[24+j]
    # for i in range(data.shape[0]):
    if(temp>t5):
        return 1
            
    temp = data[30+j]
    # for i in range(data.shape[0]):
    if(temp>t6):
        # vpm
        return 1       
        
    temp = data[36+j]
    # for i in range(data.shape[0]):
    if(temp>t7):
        return 1
    temp = data[42+j]
    # for i in range(data.shape[0]):
    if(temp>t8):
        return 1     
        
    temp = data[48+j]
    # for i in range(data.shape[0]):
    if(temp>t9):
        return 1          

    temp = data[54+j]
    # for i in range(data.shape[0]):
    if(temp>t10):
        return 1    
            
    temp = data[60+j]
    # for i in range(data.shape[0]):
    if(temp>t11):
        return 1            
    temp = data[66+j]
    # for i in range(data.shape[0]):
    if(temp>t12):
        return 1         
            
    return 0
    
def cc():
    data= np.array([1,2,3])
    vec =  np.array([1,2,3])
    print(vec.shape)
    res = classit1(data,vec)
    print(res)
    
def get_nor():
    ii =1
    path1 ="../stat/"
    df1 = pd.read_csv(path1+"non_"+str(ii)+'.csv')
    
    for ii in range(2,8):
        df2 = pd.read_csv(path1+"non_"+str(ii)+'.csv')
        df1 = pd.concat([df1,df2])
        
    print(df1.shape)
    print(df1.head())
    return df1.values
    
def get_eve():
    ii =1
    path1 ="../stat/"

    df1 = pd.read_csv(path1+"com_"+str(ii)+'.csv')
    # dy1 = pd.read_csv(path2 + str(ii)+".csv")
    for ii in range(2,14):
        # df2 = np.load(path1+str(ii)+'.npy')
        # df1 = np.concatenate((df1, df2), axis=0)
        df2 = pd.read_csv(path1+"com_"+str(ii)+'.csv')
        df1 = pd.concat([df1,df2])
    # dy1 = dy1.values
    # dy1.label[dy1.label!=3]=0
    # dy1.label[dy1.label==3]=1
    y = df1.pop("label")
    
    print(df1.shape)
    print(y.shape)
    return df1.values,y.values
    
def get_eve2():
    ii =1
    path1 ="../stat/"
    path2 ="../dif_scal/30/"
    df1 = pd.read_csv(path1+"com_"+str(ii)+'.csv')
    dy1 = pd.read_csv(path2 + "y_S"+str(ii)+".csv")
    for ii in range(2,14):

        df2 = pd.read_csv(path1+"com_"+str(ii)+'.csv')
        df1 = pd.concat([df1,df2])
        dy2 = pd.read_csv(path2 + "y_S"+str(ii)+".csv")
        dy1 = pd.concat([dy1,dy2])
    # dy1 = dy1.values
    dy1.label[dy1.label!=1]=0
    # dy1.label[dy1.label==]=1
    df1.pop("label")
    y =  np.squeeze(dy1)
    
    print(df1.shape)
    print(y.shape)
    return df1.values,y.values 

    
def fun3(v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,
         i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,
        f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,
        r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,
        a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,
        re1,re2,re3,re4,re5,re6,re7,re8,re9,re10,re11,
        re12,re13,re14
        ):
    
    
    vec = np.array([ 
    v1,i1,f1,r1,a1,re1,
    v2,i2,f2,r2,a2,re2,
    v3,i3,f3,r3,a3,re3,
    v4,i4,f4,r4,a4,re4,
    v5,i5,f5,r5,a5,re5,
    v6,i6,f6,r6,a6,re6,
    v7,i7,f7,r7,a7,re7,
    v8,i8,f8,r8,a8,re8,
    v9,i9,f9,r9,a9,re9,
    v10,i10,f10,r10,a10,re10,
    v11,i11,f11,r11,a11,re11,
    v12,i12,f12,r12,a12,re12,
    v13,i13,f13,r13,a13,re13,
    v14,i14,f14,r14,a14,re14,    
    
    ])
    
    pred =[]
    real = []
    for i in range(dt1.shape[0]):
        temp = dt1[i]
        real.append(0)
        pred.append(classit1(temp,vec))
    
    for i in range(dt2.shape[0]):
        temp = dt2[i]
        real.append(dt3[i])
        pred.append(classit1(temp,vec))    
    
    
    acc = accuracy_score(pred,real)
    print(acc)
    return acc    
    

from bayes_opt import BayesianOptimization    

def main():
    s1 = timeit.default_timer()


    global dt1,dt2,dt3
    dt1 = get_nor()
    dt2,dt3 = get_eve2()
    
    # Bounded region of parameter space
    pbounds2 = {
                'v1':(0.01,100),
                'v2':(0.01,50),
                'v3':(0.01,30),
                'v4':(0.01,15),
                'v5':(0.01,15),
                'v6':(0.01,15),
                'v7':(0.01,10),
                'v8':(0.01,8),
                'v9':(0.01,6),
                'v10':(0.01,6),
                'v11':(0.01,6),
                'v12':(0.01,2),
                'v13':(0.01,2),
                'v14':(0.01,1),
                                
                'i1':(0.01,50),
                'i2':(0.01,30),
                'i3':(0.01,20),
                'i4':(0.01,12),
                'i5':(0.01,15),
                'i6':(0.01,15),
                'i7':(0.01,10),
                'i8':(0.01,8),
                'i9':(0.01,6),
                'i10':(0.01,6),
                'i11':(0.01,6),
                'i12':(0.01,2),
                'i13':(0.01,1),
                'i14':(0.01,0.6),
                
                'f1':(0.01,10),
                'f2':(0.01,6),
                'f3':(0.01,4),
                'f4':(0.01,3),
                'f5':(0.01,2),
                'f6':(0.01,2),
                'f7':(0.01,2),
                'f8':(0.01,1.5),
                'f9':(0.01,1),
                'f10':(0.01,1),
                'f11':(0.01,0.75),
                'f12':(0.01,0.6),                
                'f13':(0.01,0.75),
                'f14':(0.01,0.15),  
                
                'r1':(0.01,100),
                'r2':(0.01,50),
                'r3':(0.01,30),
                'r4':(0.01,15),
                'r5':(0.01,15),
                'r6':(0.01,15),
                'r7':(0.01,10),
                'r8':(0.01,8),
                'r9':(0.01,6),
                'r10':(0.01,6),
                'r11':(0.01,6),
                'r12':(0.01,2),
                'r13':(0.001,0.1),
                'r14':(0.001,0.05),

                
                'a1':(0.01,40),
                'a2':(0.01,20),
                'a3':(0.01,15),
                'a4':(0.01,8),
                'a5':(0.01,8),
                'a6':(0.01,8),
                'a7':(0.01,6),
                'a8':(0.01,5),
                'a9':(0.01,5),
                'a10':(0.01,4),
                'a11':(0.01,4),
                'a12':(0.01,2),
                'a13':(0.01,0.6),
                'a14':(0.01,0.3),

                
                're1':(0.01,50),
                're2':(0.01,30),
                're3':(0.01,20),
                're4':(0.01,15),
                're5':(0.01,10),
                're6':(0.01,10),
                're7':(0.01,8),
                're8':(0.01,6),
                're9':(0.01,6),
                're10':(0.01,6),
                're11':(0.01,6),
                're12':(0.01,3),
                
                're13':(0.01,2),
                're14':(0.01,1)   
                
                }
    optimizer = BayesianOptimization(
        f=fun3,
        pbounds=pbounds2,
        random_state=int(sys.argv[1]),
    )
    optimizer.maximize(
        init_points=20,
        n_iter=180
    )
    tt = str(optimizer.max)
    with open('trans_all.txt','a+') as f:    #设置文件对象
        f.write(tt+'\n')                 #将字符串写入文件中
    print(tt)        
    s2 = timeit.default_timer()  
    print ('Runing time is Hour:',round((s2 -s1)/3600,2))
        

if __name__ == '__main__':  

    main()
    # get_eve()
    # get_eve2()
