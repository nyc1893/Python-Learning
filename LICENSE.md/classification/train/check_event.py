# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:47:16 2020

@author: iniazazari

This code show extracted data event list
"""
##calculating running time


## importing libariries
import pandas as pd
import numpy as np
# import pyarrow.parquet as pq
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle
import os
# import cv2
import timeit


path1 = '../8weeks_2_interpolated/'

rootpath = os.listdir(path1)
print(type(rootpath))
rootpath.sort(key= lambda x:str(x))
a = {}


for i in range(0,len(rootpath)):
    c = rootpath[i].split(".")[0]
    rootpath[i]= c.split("_B")[0]
import re
# from collections import Counter
# print(Counter(rootpath))
for i in rootpath:
    if rootpath.count(i)!=23:
        a[i] = rootpath.count(i)
k = str(a)

k2= re.split('[\'*]+',k)
ln = list(filter(lambda s:isinstance(s,str) and len(s) > 14,k2))
# print(type(k2))
print(ln)
for i in range(0,len(ln)):
    print(ln[i])
    
    
