# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:47:16 2020

@author: Yunchuan

This code extract the event information based on Event and related PMU from Eventdet Alg.
"""
##calculating running time


## importing libariries
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys

from sklearn.model_selection import train_test_split
from datetime import datetime
# import cv2
import timeit
def get_time(String):
    year = String.split("_")[0]
    day = String.split("_")[2]
    num = int(String.split("_")[3])

    if String.split("_")[1] == 'Jan':
        month = 1
    elif String.split("_")[1] == 'Feb':
        month = 2
    elif String.split("_")[1] == 'Mar':
        month = 3
    elif String.split("_")[1] == 'Apr':
        month = 4
    elif String.split("_")[1] == 'May':
        month = 5 
    elif String.split("_")[1] == 'Jun':
        month = 6
    elif String.split("_")[1] == 'Jul':
        month = 7  
        
    elif String.split("_")[1] == 'Aug':
        month = 8
    elif String.split("_")[1] == 'Sep':
        month = 9
    elif String.split("_")[1] == 'Oct':
        month = 10
    elif String.split("_")[1] == 'Nov':
        month = 11 
    elif String.split("_")[1] == 'Dec':
        month = 12
           
    # print(year,month,day,num)
    String = str(month) +'/'+day+'/'+ year
    return num,String

    
    
def get_class(file_name,String):    

    data = pd.read_csv(file_name)
    data['Start'] = pd.to_datetime(data['Start'])
    data = data.set_index('Start')
    num, str =get_time(String)
    dt = data[str]
    # print(dt.iloc[num,2])
    return dt.iloc[num,2]
start = timeit.default_timer()


list_pmu = [
'B126',
'B161',
'B176',
'B193',
'B232',
'B326',
'B328',
'B450',
'B457',
'B462',
'B500',
'B513',
'B623',
'B641',
'B703',
'B750',
'B780',
'B789',
'B885',
'B890',
'B904',
'B968',
'B992']
    

event_log = '../../B_T.csv'


def save_splitdata(num):
    p1 = open('X'+str(num)+'_rocof.pickle',"rb")
    X_train= pickle.load(p1)
    print(X_train.shape)

    p2 = open('y'+str(num)+'_rocof.pickle',"rb")
    y= pickle.load(p2)

    y = y[:,0]
    print(type(y))
    print(y.shape)
    y3 = np.unique(y)
    print(y3.shape)
    a = np.arange(0,y3.shape[0])
    tr,val = train_test_split(a,test_size=0.25)   
    print(tr.shape)
    print(val.shape)
    path2 = 'index/'
    np.save(path2+'tr_'+str(num)+'.npy',tr) 
    np.save(path2+'val_'+str(num)+'.npy',val)
    
def load_data(num):

    path2 = 'index/'
    tr=np.load(path2 +'tr_' +str(num)+'.npy')
    val=np.load(path2 +'val_' +str(num)+'.npy')    
    
    p1 = open('X'+str(num)+'_rocof.pickle',"rb")
    p1= pickle.load(p1)
    p1 = pd.DataFrame(p1)

    p2 = open('y'+str(num)+'_rocof.pickle',"rb")
    y= pickle.load(p2)
    y2 = y[:,0]

    df = pd.DataFrame(y)
    df.columns = ['a','b']
    y3 = np.unique(y2)
    ytr = y3[tr]
    ytest = y3[val]
    ytest =  ytest.tolist()
    print(type(ytest))
    # print(ytr[0:5])
    print(ytest[0:5])
    print(df.shape)
    df1 = df[df['a'].isin(ytr)]
    df2 = df[df['a'].isin(ytest)]
    
    
    ind1 = df1.index.tolist()
    ind2 = df2.index.tolist()
    X_train = p1.loc[ind1]
    X_test = p1.loc[ind2]
    print(X_train.shape)
    print(df1.shape)
    
    print(X_test.shape) 
    print(df2.shape)    
    return X_train,df1,X_test,df2
    
def main():
    s1 = timeit.default_timer()  
    num = 3
    load_data(num)
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
    
if __name__ == '__main__':  


    # num = int(sys.argv[1])

    main()





   


    
