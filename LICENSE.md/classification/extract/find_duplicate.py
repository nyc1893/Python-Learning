# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:47:16 2020

@author: iniazazari

This code converts the events based on their features and locations to a 2D image
"""
##calculating running time


## importing libariries
import pandas as pd
import numpy as np
import os
import timeit

df = pd.read_csv('B_Training1.csv')
flag =0
if flag ==1:
    df =  df.fillna('0')
    res = df[df['Cause'].str.contains("Planned")]
    ind = res.index
    print(df.shape)
    df = df.drop(ind)

    print('Unplanned:',df.shape)
    print('planned:',res.shape)





df['value']  = 1


# df.groupby(["Name", "City"], as_index=False)['Val'].count()
dt =  df.groupby(df['StartTime'], as_index=False)['value'].count()
dt = dt[dt['value']!=1]

print('duplicate number: ',dt['value'].sum())
print(dt.shape)
print(dt.head(20))
    
"""
"""
