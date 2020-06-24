# -*- coding: utf-8 -*-
"""
This code check the NAN data after interpolation
and show each line that contains the number
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

# min-max normalize each of PMU mesurements ([0 1])
# def normalize(x):
#     x=( x-np.min(x) )/ ( np.max(x)-np.min(x) )
#     return x
    

#Constants
num_time_sample=36000
num_pmu=23

#loading event log
event_log = 'B_T.csv'


## loading PMU names

PMU_list = pd.read_excel('PMU_reporting_rate.xlsx', sheet_name='Sheet1')
PMU_list_1 = PMU_list['60fps']


#features and labels
features=[] #array of location and time of all the events
labels=[]
missing_event=[] # save name of events that had missing values in their PMUs

##reading events one by one
count=0 #count number of matrix that are not in the specified size
iteration=-1
# path1 = '../cc/'
path1 = '../8weeks_2_interpolated/'
# with os.scandir('/Volumes/Iman/IBM data/Events/B/Original/8 weeks-1/') as entries:
j = 0
min = num_time_sample
st=[]
rootpath = os.listdir(path1)
rootpath.sort(key= lambda x:str(x))
sum = 0
for i in range(1,23*295+1):
    

    filename = path1 + rootpath[i-1]
    
    if os.path.exists(filename) ==1:
        df=pq.read_table(filename).to_pandas()
        df = df['df']
        
        if (df.isna().sum()!=0):
            print('filename = ',rootpath[i-1])
            with open("log_2empty.txt","a") as f:
                f.write("\n")
                f.write(str(df.isna().sum()))            
            sum = sum + df.isna().sum()
            

    elif os.path.exists(filename) ==0:
       print('fail to read')
    if((i)%23==0 and i!=1):
        j = j+1
        print('j = ',j)
        

        






   


    
