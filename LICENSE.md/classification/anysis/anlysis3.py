#Get distribution for different  PMUs

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




def main(jj):
    b = [0]*23

    
    list = ['Line','XFMR','Frequency','Oscillation']
    s = list[jj]
    fileList=os.listdir(path1)
    for f in fileList:

        if s in f:    
            # print(f)
            df = pd.read_excel(path1+f)

            if df.shape[0] ==1:
                for j in range(0,23):
                    if(df['pmu'][0] ==list_pmu[j]):
                        b[j]=b[j]+1
            else:
                for i in range(0,df.shape[0]):
                    for j in range(0,23):
                        if(df['pmu'][i] ==list_pmu[j]):
                            b[j]=b[j]+1
    print(b)

    x = np.arange(len(b))  # the label locations
    width = 0.5  # the width of the bars
    plt.rcParams['figure.figsize'] = (4.5,3)
    fig, ax = plt.subplots()
    
    rects1 = ax.bar(x , b, width)
    # rects2 = ax.bar(x + width/2, b, width, label='6weeks')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(str(s)+' events count vs PMU in S'+str(k))
    ax.set_ylabel('count')
    # ax.set_title('遗忘率柱状图')
    ax.set_xticks(x)
    # ax.set_rotation(50)
    ax.set_xticklabels(list_pmu,rotation = 90)
    # plt.ylim(0, 90)
    # plt.figure( figsize=(4.5,3))
    plt.tight_layout()
    plt.savefig('f'+str(jj)+'.png') 
    # plt.show()
    # ax.legend()
        
def hh(j):

    Time=[]
    fileList=os.listdir(path1)
    list = ['Line','XFMR','Frequency','Oscillation']
    s = list[j]

    for f in fileList:

        if s in f:    
            df = pd.read_excel(path1+f)
            df['time'] = df['time'].map(lambda x: x.lstrip("['").rstrip("']"))
            df['dif'] = pd.to_datetime(df['time']) - pd.to_datetime(df['start_time'])
            

            # print(datetime.strptime(str(df['dif'][0]), 'd%days %H:%M:%S'))
            for i in range(0,df.shape[0]):
                # Time.append(datetime.strptime(df['dif'][i], '%M:%S'))
                Time.append(round(df['dif'][i].total_seconds()))
    print(len(Time))  
    plt.figure( figsize=(4.5,3))
    plt.savefig('f'+str(j)+'.png') 
    # plt.show()
if __name__ == '__main__':  
    for jj in range(0,3+1):
        main(jj)
    # main(3)    

    
    
    
    # nohup python -u EventDet.py > test.log 2>&1 &    
