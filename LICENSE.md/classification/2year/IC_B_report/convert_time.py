# -*- coding: utf-8 -*-
"""
Created on 9/15/2021  
This one generate the finial event log, 
and also convert the difference time from ind to date_time format
@author: Barron
"""
# 

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import math
from datetime import datetime, timedelta
def fun():
    df= pd.read_csv("B_T2.csv")
    df['ind'].loc[df['type']==0] = (df['ind']+60 -18000)/60
    df['ind'].loc[df['type']==2] = (df['ind']+300 -18000)/60
    print(df.dtypes)
    df['F_time']=pd.to_datetime(df['Start'], format='%Y/%m/%d %H:%M')
    print(df.dtypes)
    
    
    
 
    
    res =[]
    df['ind']=df['ind'].fillna(0)
    det =[]
    reson =[]
    # treat all as normal
    
    for i in range(df.shape[0]):
        temp = df.loc[i,"F_time"] +timedelta(seconds = df.loc[i,"ind"])
        res.append(temp)
        if(df.loc[i,"detected"] == 1):
            det.append("Yes")
            reson.append(" ")
        else:
            det.append("No")
            if(df.loc[i,"ind"] == 0):
                reson.append("Data Unavailable")
            else:
                reson.append(" ")
                
    df["c_time"] = res
    df["det"] = det
    df["reason"] = reson
    # df['change']= df['change']
    df['c_time'] = df['c_time'].dt.strftime('%Y/%m/%d %H:%M:%S.%f')

    # revise the result that is close to mid-night

    for i in range(df.shape[0]):
        temp = df.loc[i,"F_time"] +timedelta(seconds = df.loc[i,"ind"])


    print(df.dtypes)
    print(df.head())
    df.to_csv("B_T3.csv",index =0)
    
    
def revs():
    df= pd.read_csv("B_T3.csv")
    dt = df.loc[0,"F_time"]
    print(dt)
    x = dt.split(" ")[1].split(":")
    print(x)    
    
def fun2():
    df= pd.read_csv("B_T3.csv")
    df.reason.loc[(df["detected"] == 0) ] = " "
    df.to_csv("B_T3.csv",index =0)
    
    dt = df[df["detected"] == 0]
    print(dt.shape)
    print(dt)
def fun3():
    df= pd.read_csv("B_T3.csv")
    ll = ["F_time","Category","Cause","Descriptor","det","c_time","reason"]
    df = df[ll]

    df.to_csv("B_T4.csv",index =0)
    
    
def fun4():
    df= pd.read_csv("B_T3.csv")
    dt = df[df["detected"] != 1]


#revise ind value
def revs():
    df= pd.read_csv("B_T2.csv")
    res =[]
    for i in range(df.shape[0]):
        dt = df.loc[i,"Start"]
        # print(dt)
        x = dt.split(" ")[1].split(":")
        for j in range(2):
            x[j] = int(x[j])

        if((x[0]==23 and x[1]>55) or (x[0]==0 and x[1]<5) ):
        
            print(dt)
            res.append(i)

    print(len(res))
def main():
    # fun()
    # fun2()
    # fun3()
    # fun4()
    
    
    revs()
if __name__ == '__main__':  
    main()
