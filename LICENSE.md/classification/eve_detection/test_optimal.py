

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

from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score,classification_report

   
def classit2(data,t1,t3,t5,t7,t8):

    # list5 =[0,2,4,6,7]
    
    temp = data[:,0]
    for i in range(temp.shape[0]):
        if(temp[i]>t1):
            # vpm
            return 1

            
    temp = data[:,2]
    for i in range(temp.shape[0]):
        if(temp[i]>t3):
            # vpm
            return 1
      
            
    temp = data[:,4]
    for i in range(temp.shape[0]):
        if(temp[i]>t5):
            # vpm
            return 1
            

            
    temp = data[:,6]
    for i in range(temp.shape[0]):
        if(temp[i]>t7):
            # vpm
            return 1               
    # print("temp.shape",temp.shape)
    temp = data[:,7]
    for i in range(temp.shape[0]):
        if(temp[i]>t8):
            # vpm
            return 1   
            
            
    return 0
   
def get_index():
    ii =1
    path1 ="../stat/non_S_"
    res =[]
    df=[] 
    df2=[]
    for ii in range(1,13+1):
        df1 = np.load(path1+str(ii)+'.npy')
        temp=[ii]*df1.shape[0]
        res.append(temp)
        for i in range(df1.shape[0]):
            df2.append(i)

    df = [i for k in res for i in k]
    df =pd.DataFrame(df)
    df.columns =["s"]
    df["NO"] = df2
    print(df.head())
    df.to_csv("cc.csv",index=None)
   
def get_nor():
    ii =1
    path1 ="../stat/non_S_"
    df1 = np.load(path1+str(ii)+'.npy')
    # list_s=[1]*df1.shape[0]
    for ii in range(2,14):
        df2 = np.load(path1+str(ii)+'.npy')
        df1 = np.concatenate((df1, df2), axis=0)
        # list_s.append([ii]*df2.shape[0])
    # print(df1.shape)
    # print(len(list_s))
    return df1
    
def get_eve():
    ii =1
    path1 ="../stat/event_S_"
    df1 = np.load(path1+str(ii)+'.npy')
  
    for ii in range(2,14):
        df2 = np.load(path1+str(ii)+'.npy')
        df1 = np.concatenate((df1, df2), axis=0)
    # print(df1.shape)
    return df1
    

    
def read(ii):
    result=[]
    with open('tr+test.txt','r') as f:
        for line in f:
            result.append(list(line.strip('\n').split(',')))
    cc = str(result[ii])
    x = cc.split(': ')
    # print(x)

    t1 = float(x[3].split('\",')[0])
    t3 = float(x[4].split('\",')[0])
    t5 = float(x[5].split('\",')[0])
    t7 = float(x[6].split('\",')[0])    
    t8 = float(x[7].split('\",')[0].split('}')[0])   
    # print(x[4].split('\",')[0])
    # print(x[5].split('\",')[0])
    # print(x[6].split('\",')[0])
    # print(x[7].split('\",')[0].split('}')[0])   
    print(t1,t3,t5,t7,t8)
    return t1,t3,t5,t7,t8
    
def fun2(j,dt1,dt2,t1,t3,t5,t7,t8):

    path2 ="index2/"
    arr = np.load(path2+'bo.npy')
    # j = 0
    num = round(0.2*arr.shape[0])
    test_index = list(arr[j*num:num*(j+1)])

    train_index=list(set(arr)-set(test_index))
    pred2 =[]
    real2 = []    
    
    pred =[]
    real = []
    for i in train_index:
        temp = dt1[i]
        real.append(0)
        pred.append(classit2(temp,t1,t3,t5,t7,t8))
    
    for i in train_index:
        temp = dt2[i]
        real.append(1)
        pred.append(classit2(temp,t1,t3,t5,t7,t8))    
        
    for i in test_index:
        temp = dt1[i]
        real2.append(0)
        pred2.append(classit2(temp,t1,t3,t5,t7,t8))
    
    for i in test_index:
        temp = dt2[i]
        real2.append(1)
        pred2.append(classit2(temp,t1,t3,t5,t7,t8))    
    
    print("traning...")
    acc = accuracy_score(pred,real)
    # print(acc)
    f1 = f1_score(pred,real, average='weighted') 
    # print(f1)
    # df =pd.DataFrame(pred)
    # print(df.shape[0])
    
    matrix=confusion_matrix(pred,real)
    print(matrix)    
    
    class_report=classification_report(pred,real)
    print(class_report)    
    
    print("test...")
    acc = accuracy_score(pred2,real2)
    print(acc)
    f1 = f1_score(pred2,real2, average='weighted') 
    print(f1)
    df =pd.DataFrame(pred2)
    matrix=confusion_matrix(pred2,real2)
    print(matrix)    
    
    class_report=classification_report(pred2,real2)
    print(class_report)    
    
    
    # df.columns =["pred"]
    # df["real"] = real2
    # df2 = pd.read_csv("ck.csv")
    # df["s"] = df2["s"]
    # df["NO"] = df2["NO"]
    # df.to_csv("debug2.csv",index= None)
    
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
        

    # print("traning...")
    acc = accuracy_score(pred,real)
    # print(acc)
    f1 = f1_score(pred,real, average='weighted') 
    # print(f1)
    df =pd.DataFrame(pred)
    print(df.shape[0])
    df.columns =["pred"]
    matrix=confusion_matrix(pred,real)
    print(matrix)    
    
    class_report=classification_report(pred,real)
    print(class_report)    
    
    df["real"] = real
    df2 = pd.read_csv("ck.csv")
    df["s"] = df2["s"]
    df["NO"] = df2["NO"]
    df.to_csv("debug3.csv",index= None)
    
    
    
def check_error():
    df3=pd.read_csv("debug.csv")
    
    ind= df3[df3["pred"]!=df3["real"]].index
    df3 = df3.iloc[ind] 
    print(df3.head())
    print(ind)
    df3.to_csv("err.csv",index =None)    


        
def tt():
    global dt1,dt2
    dt1 = get_nor()
    dt2 = get_eve()
    
    # for i in range(50):
        # t1,t3,t5,t7,t8 = read(0)
        # j = i // 10
        # print("j=",j)
        # fun2(j,dt1,dt2,t1,t3,t5,t7,t8)
    # t1,t3,t5,t7,t8 = read(49)
    t1 = 0.2
    t3 = 0.5
    t5 = 0.05
    t7 = 0.4
    t8 = 0.3
    
    fun3(t1,t3,t5,t7,t8)
    print(dt1.shape)
    print(dt2.shape)
    check_error()
    
def rd_eve_type(): 
    sum = 0
    path4 = "../dif_scal/30/"
    df = pd.read_csv(path4+"y_S"+str(1)+".csv")
    for i in range(2,14):
        df2 = pd.read_csv(path4+"y_S"+str(i)+".csv")
        df = pd.concat([df,df2])
    print(df.shape)
    df.columns =["label"]
    df.to_csv("ccc.csv",index = 0)
    
def replot():
    df = pd.read_csv("debug3.csv")
    df = df.fillna(-1)
    print(df.head())
    dt1 = df[df["label"]== 1]
    dt2 = df[df["label"]== -1]
    dt3 = pd.concat([dt1,dt2])
    
    pred = dt3["pred"]
    real = dt3["real"]
    
    matrix=confusion_matrix(pred,real)
    print(matrix)    
    
    class_report=classification_report(pred,real)
    print(class_report)    
    
    
def test_sub_opt():
    global dt1,dt2
    dt1 = get_nor()
    dt2 = get_eve()
    t1 = 0.7
    t3 = 0.8
    t5 = 1
    t7 = 0.1
    t8 = 0.1
    # t1,t3,t5,t7,t8 = read(49)
    fun3(t1,t3,t5,t7,t8)
    
def main():
    s1 = timeit.default_timer()  
    # rd_eve_type()
    # tt()
    # replot()
    # test_sub_opt()
    s2 = timeit.default_timer()  
    print ('Runing time is Hour:',round((s2 -s1)/3600,2))
    
if __name__ == '__main__':  

    main()
    # get_nor()
