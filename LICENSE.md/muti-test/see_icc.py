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
import timeit

def main():
    s1 = timeit.default_timer()  
    file_name = 'C_Training.csv'
    df = pd.read_csv(file_name)
    df =df.fillna("No_event type")
    # print(df.head())
    # a = df['Event'].values
    # df['id'].value_counts()
    print(df['Category'].value_counts() )
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
    
if __name__ == '__main__':  


    # num = int(sys.argv[1])

    main()

