#Get histogram distribution by time

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
k = 6
path1 = 'det'+str(k)+'/'
# fname = '2016_Feb_01_0_Line Outage'
# df=pq.read_table(path1+fname).to_pandas()




# import xlrd
    

def main():    

    fileList=os.listdir(path1)
    for f in fileList:
        
        if 'Osci' in f:
            print(f)
        elif 'Line' in f:
            print(f)            
            
        # x = f.split(".", 1)
        # print(x[0])
        # if x[0]:
            # fname = x[0]
            # df=pq.read_table(path1+fname+'.parquet').to_pandas()
            # deal(df,fname)
        # else:
            # continue
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
    fileList=os.listdir(path1)
    list = ['Line','XFMR','Frequency','Oscillation']
    s = list[j]

    for f in fileList:
        
        # if 'Osci' in f:
        if s in f:    
            df = pd.read_excel(path1+f)
            df['time'] = df['time'].map(lambda x: x.lstrip("['").rstrip("']"))
            df['dif'] = pd.to_datetime(df['time']) - pd.to_datetime(df['start_time'])
            
            # print(df.shape)
            # print(df['dif'][0].total_seconds())
            
            # print(datetime.strptime(str(df['dif'][0]), 'd%days %H:%M:%S'))
            for i in range(0,df.shape[0]):
                # Time.append(datetime.strptime(df['dif'][i], '%M:%S'))
                Time.append(round(df['dif'][i].total_seconds()))
    print(len(Time))  
    # print((Time))    
    # a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27]) 
    # data = randn(20)
    # print(data)
    plt.figure( figsize=(4.5,3))
    plt.hist(Time, bins=30, color=sns.desaturate("indianred", .8), alpha=.4)  
    plt.title('Histgram in S'+str(k)+' '+s + ' events : '+str(len(Time)))
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
