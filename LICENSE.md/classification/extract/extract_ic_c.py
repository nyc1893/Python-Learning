
# -*- coding: utf-8 -*-
"""
extract 10 mins data based on Eventlog C
The file should located in scrach/share/ic_c/y2017_m5
Single thread
@author: Barron
"""

import multiprocessing
import pandas as pd
import numpy as np
from functools import partial
import pyarrow.parquet as pq
import glob
from datetime import datetime, timedelta
import time
import sys
import os
            
def Extract_event(filename,start_time,savename):

    # df2 = pd.read_csv(file2)
    df = pd.read_parquet(filename)
    df = df.sort_index()
    

    # print(df.shape)

    report_time = start_time
    # report_time = datetime.strptime(start_time, '%Y/%m/%d  %H:%M:%S') 

    s_time=report_time-timedelta(minutes = 5)
    e_time=report_time+timedelta(minutes = 5)
    dc = df[s_time:e_time]
    # dc['new'] = df['f'].shift(periods=1)
    # dc['df']  =dc['f']-dc['new'] 
    dc = dc.iloc[1:]
    # print(dc.shape)
    # print(dc.head())
    dc.to_parquet(savename)    
    # print("save done")    
    
    

def main():
    start = time.time()
    # multiprocessing.freeze_support() 
    findyear = 2017
    findmonth = 5
    findday= int(sys.argv[1])
    findtime = str(findyear)+str("-")+str(findmonth)+str("-")+str(findday)
    
    # findtime= datetime.strptime(findtime, '%Y-%m-%d')
    
    file2 = "../../C_Training.csv"
    df = pd.read_csv(file2)
    df['StartTime'] = pd.to_datetime(df['StartTime'])
    df.set_index("StartTime", inplace=True)
    df = df.sort_index()
    
    dt = df[findtime]
    dt['time'] = dt.index
    print(dt.head())
    # print(dt)
    dt =dt.values
    for i in range(dt.shape[0]):
        start_time = dt[i,4]
        event_type = dt[i,1]
        # print(start_time)
        # print(event_type)
        do((start_time),event_type,i)
        
def do(start_time,event_type,num):    
    
        # start_time = "2017/5/2  6:59:02"
        # event_type = "Line"
    report_time = start_time
    # datetime.strptime(start_time, '%Y/%m/%d  %H:%M:%S') 
    list = ['','Jan','Feb','Mar','Apr','May','Jun',
            'Jul','Aug','Sep','Oct','Nov','Dec']
    yy = report_time.year
    mm = list[report_time.month]
    dd = report_time.day

    path = "/home/ycliu/box2/"
    path2= path + "Positive_Sequence/theMonth="+str(report_time.month)+ "/theDay="+str(report_time.day)
    fileList=os.listdir(path2)
    for f in fileList:    
        x = f.split(".parquet")
        print(x[0]+".parquet")
        file = path2+"/"+x[0]+".parquet"
        savename = str(yy)+"_"+mm+"_"+str(dd)+"_"+str(num)+"_"+event_type+"_"+str(x[0])+".parquet"
        # pool = multiprocessing.Pool(40)
        # pool.apply_async(Extract_event, args=(file,start_time,savename)) 
        Extract_event(file,start_time,savename)
            # pool.close()
            # pool.join()  
 
    # print(type(yy))
    # event_num = 1
    
    
    # print(savename)
    # Extract_event(file,start_time,savename)
    
    print("Mutiprocessing time: {}mins\n".format((time.time()-start)/60))
    print("Mutiprocessing time: {}secs\n".format((time.time()-start)))

        
if __name__ == "__main__":
    main()
