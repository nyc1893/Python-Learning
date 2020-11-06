# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:47:16 2020

@author: Yunchuan

This code extract the event information based on Event and related PMU 
from the excel files of Eventdetection result.
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
    

event_log = '../../B_T.csv'

def test(num):

    path1 = '../hr1/det'+str(num)+'/'
    path2 = '../../../processed/8weeks_'+str(num)+'_inter/'   
    st=[]
    labels =[] 
    ename = []

    rootpath = os.listdir(path1)
    # rootpath.sort(key= lambda x:str(x))
    
    filename = open('a.txt','w')
    for value in rootpath:
        filename.write(str(value)+'\n')
    filename.close()
    nn = len(rootpath)
    print('len(rootpath)',nn)
    
    fps=60
    start_crop=int(fps*60*4)
    stop_crop=int(fps*60*7)
    for j in range(0,nn):
        if '.xls' in rootpath[j]:
            filename = path1 + rootpath[j]
            
        # printZ"('i = ',i")
            if os.path.exists(filename) ==1:
                # fname = '2016_Feb_01_1_Line Outage.xls'
                df = pd.read_excel(filename)
                p_list =[]
                # print(df.head())
                for i in range(0,df.shape[0]):
                    p_list.append(df['pmu'][i])
                # print(p_list)
                x = rootpath[j].split(".")

                # print(x[0])
                planned_event=['Planned Operations','Planned Service', 'Planned Testing']
                for i in range(0,df.shape[0]):
                    fname2 = x[0]+'_'+p_list[i]
                    print(fname2)
                    if os.path.exists(path2+fname2+'.parquet') ==1:
                        df2=pq.read_table(path2+fname2+'.parquet').to_pandas()
                        st.append(df2['ip_m'][start_crop:stop_crop])  #get the 1st 1 mins data for X
                        # print(df2.head())
                        event_name = x[0]
                        ename.append(event_name)
                        
                        # get the y label
                        if 'Line' in event_name:
                            temp=  get_class(event_log,event_name)
                            if temp==planned_event[0] or temp==planned_event[1] or temp==planned_event[2]:
                               labels.append(4)
                            else:
                                labels.append(0)
                                    
                        elif 'XFMR' in event_name:
                            temp=  get_class(event_log,event_name)
                            if temp==planned_event[0] or temp==planned_event[1] or temp==planned_event[2]:
                               labels.append(5)
                            else:
                                labels.append(1)
                                  
                        elif 'Frequency' in event_name: 
                            labels.append(2)
                        elif 'Oscillation' in event_name: 
                            labels.append(3)   
                
       
    st =   np.array(st) 
    labels = np.array(labels)
    
    print('st.shape',st.shape)
    pickle_out = open('X'+str(num)+'_ip_m2.pickle',"wb")
    pickle.dump(st, pickle_out, protocol=2)
    pickle_out.close() 
    
    num_samples = st.shape[0] 
    labels = labels.reshape(num_samples,-1)
    b = np.array(ename)
    b = b.reshape(num_samples,-1)
    labels=np.concatenate((b, labels), axis=1)
    print('labels.shape',labels.shape)
    print(labels[0:5])
    
    pickle_out = open('y'+str(num)+'_ip_m2.pickle',"wb")
    pickle.dump(labels, pickle_out, protocol=2)
    pickle_out.close()  
    
    
def load_data(num):
    p1 = open('X'+str(num)+'_rocof.pickle',"rb")
    X_train= pickle.load(p1)
    print(X_train.shape)

    p2 = open('y'+str(num)+'_rocof.pickle',"rb")
    y= pickle.load(p2)
    y2 = pd.DataFrame(y)
    y2.to_csv('yy.csv',index =None)
    y = y[:,0]
    y3 = np.unique(y)
    print('len(np.unique(y))',len(np.unique(y)))
    filename = open('b.txt','w')
    for value in y3:
        filename.write(str(value)+'\n')
    filename.close()
    
def main(num):
    s1 = timeit.default_timer()  
    test(num)
    # load_data(num)
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
    
if __name__ == '__main__':  


    num = int(sys.argv[1])

    main(num)





   


    
