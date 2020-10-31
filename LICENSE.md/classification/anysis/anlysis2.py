

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import math
from scipy.stats.distributions import chi2
import os
from datetime import datetime

import seaborn as sns
from numpy.random import randn

for k in range(1,7):
    locals()['p'+str(k)] = 'det'+str(k)+'/'




def test():

    fname = '2016_Jan_22_4_XFMR Outage.xls'
    df = pd.read_excel(path1+fname)
    df['time'] = df['time'].map(lambda x: x.lstrip("['").rstrip("']"))
    df['dif'] = pd.to_datetime(df['time']) - pd.to_datetime(df['start_time'])
    Time=[]
    print(df.shape)
    print(df['dif'][0].total_seconds())
    
    # print(datetime.strptime(str(df['dif'][0]), 'd%days %H:%M:%S'))
    for i in range(0,df.shape[0]):
        # Time.append(datetime.strptime(df['dif'][i], '%M:%S'))
        Time.append(round(df['dif'][i].total_seconds()))
    print((Time))

def main(j):

    Time=[]
    
    list = ['Line','XFMR','Frequency','Oscillation']
    s = list[j]
    

    fileList=os.listdir(p1 )
    for f in fileList:
        if s in f:    
            df = pd.read_excel(p1+f)
            df['time'] = df['time'].map(lambda x: x.lstrip("['").rstrip("']"))
            df['dif'] = pd.to_datetime(df['time']) - pd.to_datetime(df['start_time'])
            
            # print(df.shape)
            # print(df['dif'][0].total_seconds())
            
            # print(datetime.strptime(str(df['dif'][0]), 'd%days %H:%M:%S'))
            for i in range(0,df.shape[0]):
                # Time.append(datetime.strptime(df['dif'][i], '%M:%S'))
                Time.append(round(df['dif'][i].total_seconds()))
                
    fileList=os.listdir(p2 )
    for f in fileList:
        if s in f:    
            df = pd.read_excel(p2+f)
            df['time'] = df['time'].map(lambda x: x.lstrip("['").rstrip("']"))
            df['dif'] = pd.to_datetime(df['time']) - pd.to_datetime(df['start_time'])
            
            # print(df.shape)
            # print(df['dif'][0].total_seconds())
            
            # print(datetime.strptime(str(df['dif'][0]), 'd%days %H:%M:%S'))
            for i in range(0,df.shape[0]):
                # Time.append(datetime.strptime(df['dif'][i], '%M:%S'))
                Time.append(round(df['dif'][i].total_seconds()))                
                
    fileList=os.listdir(p3 )
    for f in fileList:
        if s in f:    
            df = pd.read_excel(p3+f)
            df['time'] = df['time'].map(lambda x: x.lstrip("['").rstrip("']"))
            df['dif'] = pd.to_datetime(df['time']) - pd.to_datetime(df['start_time'])
            
            # print(df.shape)
            # print(df['dif'][0].total_seconds())
            
            # print(datetime.strptime(str(df['dif'][0]), 'd%days %H:%M:%S'))
            for i in range(0,df.shape[0]):
                # Time.append(datetime.strptime(df['dif'][i], '%M:%S'))
                Time.append(round(df['dif'][i].total_seconds()))      

    fileList=os.listdir(p4 )
    for f in fileList:
        if s in f:    
            df = pd.read_excel(p4+f)
            df['time'] = df['time'].map(lambda x: x.lstrip("['").rstrip("']"))
            df['dif'] = pd.to_datetime(df['time']) - pd.to_datetime(df['start_time'])
            
            # print(df.shape)
            # print(df['dif'][0].total_seconds())
            
            # print(datetime.strptime(str(df['dif'][0]), 'd%days %H:%M:%S'))
            for i in range(0,df.shape[0]):
                # Time.append(datetime.strptime(df['dif'][i], '%M:%S'))
                Time.append(round(df['dif'][i].total_seconds()))

    fileList=os.listdir(p5 )
    for f in fileList:
        if s in f:    
            df = pd.read_excel(p5+f)
            df['time'] = df['time'].map(lambda x: x.lstrip("['").rstrip("']"))
            df['dif'] = pd.to_datetime(df['time']) - pd.to_datetime(df['start_time'])
            
            # print(df.shape)
            # print(df['dif'][0].total_seconds())
            
            # print(datetime.strptime(str(df['dif'][0]), 'd%days %H:%M:%S'))
            for i in range(0,df.shape[0]):
                # Time.append(datetime.strptime(df['dif'][i], '%M:%S'))
                Time.append(round(df['dif'][i].total_seconds()))
                
    fileList=os.listdir(p6 )
    for f in fileList:
        if s in f:    
            df = pd.read_excel(p6+f)
            df['time'] = df['time'].map(lambda x: x.lstrip("['").rstrip("']"))
            df['dif'] = pd.to_datetime(df['time']) - pd.to_datetime(df['start_time'])
            
            # print(df.shape)
            # print(df['dif'][0].total_seconds())
            
            # print(datetime.strptime(str(df['dif'][0]), 'd%days %H:%M:%S'))
            for i in range(0,df.shape[0]):
                # Time.append(datetime.strptime(df['dif'][i], '%M:%S'))
                Time.append(round(df['dif'][i].total_seconds()))                


                
    print(len(Time))  

    plt.figure( figsize=(4.5,3))
    plt.hist(Time, bins=30, color=sns.desaturate("indianred", .8), alpha=.4)  
    plt.title('Histgram in 1 year '+s + ' events : '+str(len(Time)))
    plt.ylabel('count')
    plt.xlabel('time difference with start time(s)')
    plt.tight_layout()
    plt.savefig('f'+str(j)+'.png') 
    # plt.show()
if __name__ == '__main__':  
    for j in range(0,3+1):
        main(j)
    # test()    

    
    
    
    # nohup python -u EventDet.py > test.log 2>&1 &    
