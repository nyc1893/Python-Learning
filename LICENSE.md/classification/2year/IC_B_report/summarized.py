# -*- coding: utf-8 -*-
"""
Created on 9/15/2021  
This one summarized the result, like max index of Freq and non-Freq event
Also merge the missing and previous version of data result.

@author: Barron
"""
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from functools import partial
import pyarrow.parquet as pq
import glob
from datetime import datetime, timedelta
import time
import sys
import os
import timeit
import ED_OLAP
import pickle as pickle 

def generate_data():
    a = np.random.uniform(0,0.1,(4,200))
    a = np.zeros(0,0.1,(4,200))
    print(a.shape)
    a[1,100]=10
    plt.figure( figsize=(12,4))
    for i in range(a.shape[0]):
        x = a[i,:]
        plt.plot(range(len(x)),x,zorder=0)
    plt.savefig("k")
    a = a[:,np.newaxis,:]
    temp1 = ED_OLAP.get_single_feature(a,30)
    print(temp1.shape)
    plt.figure( figsize=(12,4))
    x2 = temp1[0,:]
    print(x2.shape)
    plt.plot(range(len(x2)),x2,zorder=0)
    plt.savefig("k2")
    



def rd_zeta(ii):
    path1 = "../zeta/S_"+str(ii)+"/"
    # f_name = path1 + "2016_Jan_21_9_Line Outage.pickle"
    
    fileList=os.listdir(path1)
    f_name = []

    res =[]
    t1 = 0.033879116618481205
    t2 = 0.024773737690786245
    t3 = 2.976857879392044
    cc =[]
    for f in fileList:
        # print(f)
        if(f.endswith(".pickle") ):    
            # f_name = path1 + f
            p1 = open(path1+f,"rb")
            X = pickle.load(p1)

            temp = classit(X,t1,t2,t3)
            # print(f,l2,temp)
            f_name.append(f.split(".")[0])

            res.append(temp)
            cc.append(find_max(X))
    df = pd.DataFrame(res)
    df.columns = ["type","detected"]
    df["f_name"] = f_name
    print(df.head())
    df["max"] = cc
    df.to_csv("res/det_"+str(ii)+".csv",index = 0)


def rd_zeta_miss():
    path1 = "../zeta/miss/"
    # f_name = path1 + "2016_Jan_21_9_Line Outage.pickle"
    
    fileList=os.listdir(path1)
    f_name = []

    res =[]
    t1 = 0.033879116618481205
    t2 = 0.024773737690786245
    t3 = 2.976857879392044
    cc =[]
    for f in fileList:
        # print(f)
        if(f.endswith(".pickle") ):    
            # f_name = path1 + f
            p1 = open(path1+f,"rb")
            X = pickle.load(p1)

            temp = classit(X,t1,t2,t3)
            # print(f,l2,temp)
            f_name.append(f.split(".")[0])

            res.append(temp)
            cc.append(find_max(X))
    df = pd.DataFrame(res)
    cc = np.array(cc)
    df.columns = ["type","detected"]
    
    df["max"] = cc
    df["f_name"] = f_name
    print(df.head())
    df.to_csv("res/miss.csv",index = 0)
    
def rd_zeta2_miss():
    path1 = "../zeta_f/miss/"

    f_name =[]
    res =[]
    fileList=os.listdir(path1)
    for f in fileList:
        # print(f)
        if(f.endswith(".pickle") ):    
            # f_name = path1 + f
            p1 = open(path1+f,"rb")
            X = pickle.load(p1)

            res.append(np.argmax(X[0,:]))

            f_name.append(f.split(".")[0])
            
    df = pd.DataFrame(res)
    df.columns = ["max"]
    df["f_name"] = f_name
    print(df.head())
    df.to_csv("res/miss_freq.csv",index = 0)
    
def classit(data,t1,t2,t3):

    res = []

    temp = data[0,:]
    for i in range(temp.shape[0]):
        if(temp[i]>t1):
            res.append(0)
            # res.append(i)
            res.append(1)
            return res

            
    temp = data[1,:]
    for i in range(temp.shape[0]):
        if(temp[i]>t2):
            res.append(1)
            # res.append(i)
            res.append(1)
            return res
      
            
    temp = data[2,:]
    for i in range(temp.shape[0]):
        if(temp[i]>t3):
            res.append(2)
            # res.append(i)
            res.append(1)
            return res
    res.append(-1)
    # res.append(-1)
    res.append(0)
    return res

def find_max(data):



    temp = data[0,:]

    return np.argmax(temp)
            



def rd_zeta2(ii):
    path1 = "../zeta_f/S_"+str(ii)+"/"
    f= "2016_Feb_01_3_Frequency Event.pickle"
    f_name =[]
    res =[]
    fileList=os.listdir(path1)
    for f in fileList:
        # print(f)
        if(f.endswith(".pickle") ):    
            # f_name = path1 + f
            p1 = open(path1+f,"rb")
            X = pickle.load(p1)

            res.append(np.argmax(X[0,:]))

            f_name.append(f.split(".")[0])
            
    df = pd.DataFrame(res)
    df.columns = ["max"]
    df["f_name"] = f_name
    print(df.head())
    df.to_csv("res/det_freq_"+str(ii)+".csv",index = 0)
    
    
def merge(ii):
    path = "res/"
    df1 = pd.read_csv(path+"det_"+str(ii)+".csv")
    df2 = pd.read_csv(path+"det_freq_"+str(ii)+".csv").values
    df1.columns = ["type",	"detected","f_name","ind"]
# type	detected	f_name	max


    for i in range(df2.shape[0]):
        df1.ind.loc[df1["f_name"] == df2[i,1]] = df2[i,0]
        df1.type.loc[df1["f_name"] == df2[i,1]] = 2
    
    df1.to_csv("res/det_"+str(ii)+".csv",index = 0)
    
def merge2():
    path = "res/"
    df1 = pd.read_csv(path+"miss.csv")
    df2 = pd.read_csv(path+"miss_freq.csv").values
    df1.columns = ["type",	"detected","f_name","ind"]


    for i in range(df2.shape[0]):
        df1.ind.loc[df1["f_name"] == df2[i,1]] = df2[i,0]
        df1.type.loc[df1["f_name"] == df2[i,1]] = 2
    
    df1.to_csv("res/det_miss.csv",index = 0)
    
def remove_z(str):
    if str[0] == '0':
        return str[1]
    else:
        return str
        
def new_feature(ii):
    path = "res/"
    df1 = pd.read_csv(path+"det_"+str(ii)+".csv")

    res = []
    for i in range(df1.shape[0]):
        temp = df1.loc[i,"f_name"]
        x = temp.split("_")
        # print(x[0]+"_"+x[1]+"_"+x[2]+"_"+x[3])
        res.append(x[0]+"_"+x[1]+"_"+remove_z(x[2])+"_"+x[3])
    # print(res[:5])
    res = np.array(res)
    df1["new"] = res
    df1.to_csv("res/det_"+str(ii)+".csv",index = 0)
    
def change():
    path = "res/"
    df1 = pd.read_csv(path+"B_T.csv")
    # df['A'] = df['A'].apply(lambda _: str(_))
    df1['new'] = df1['new'].apply(lambda _: str(_))
    print(df1["new"].dtypes)

    df2 = pd.read_csv(path+"det_miss.csv")
    for ii in range(1,14):
        df3 = pd.read_csv(path+"det_"+str(ii)+".csv")
        df2 = pd.concat([df2,df3])
    df1 = pd.merge(df1, df2, how='left')
    df1.to_csv(path+"B_T2.csv",index = 0)
    
def dett():
    for i in range(1,14):
        rd_zeta(i)
        rd_zeta2(i)
        merge(i)
        new_feature(i)
        
        
def checkmissing():
    path = "res/"
    df = pd.read_csv(path+"B_T2.csv")
    dt = df[df["detected"].isnull()]
    dt.to_csv(path+"miss.csv", index = None)

def plus_z(str):
    if len(str) == 1:
        return "0"+str
    else:
        return str

def add_zero():
    path = "res/"
    df = pd.read_csv(path+"miss.csv")
    print(df.shape)
    a =[]
    for i in range(df.shape[0]):
        temp = df.loc[i,"name"]
        x = temp.split("_")
        a.append(x[0]+"_"+x[1]+"_"+plus_z(x[2])+"_"+x[3])
    a=np.array(a)
    df["new"] =a
    df.to_csv(path+"miss.csv", index = None)
    
def main():
    s1 = timeit.default_timer()

    # dett()

    # change()
    # new_feature(1)
    # add_zero()
    # rd_zeta_miss()
    # rd_zeta2_miss()
    # merge2()
    change()
    s2 = timeit.default_timer()

    print('Time: ', (s2 - s1)/60 )        
if __name__ == "__main__":
    main()
