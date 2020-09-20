# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:47:16 2020

@author: Barron
Add mutilabel info
This code converts the events based on their features and locations to a 2D image
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

# min-max normalize each of PMU mesurements ([0 1])
# def normalize(x):
#     x=( x-np.min(x) )/ ( np.max(x)-np.min(x) )
#     return x
    



def deal(k,num):
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
    # num = 2
    path1 = '../processed/8weeks_'+str(num)+'_inter/'
    # with os.scandir('/Volumes/Iman/IBM data/Events/B/Original/8 weeks-1/') as entries:
    j = 0
    min = num_time_sample
    st=[]
    ename=[]
    rootpath = os.listdir(path1)
    rootpath.sort(key= lambda x:str(x))
    flag = 1

    nn = len(rootpath)
    # nn = 23*2
    for i in range(0,nn):
        

        filename = path1 + rootpath[i]
        # print('i = ',i)
        if os.path.exists(filename) ==1:
            df=pq.read_table(filename).to_pandas()
            # print('hahas')
            if (df['df'].isna().sum()!=0):
                flag = 0
                
            if k == 0:
                st.append(df['vp_m']) #vp_m
            elif k == 1:
                st.append(df['vp_a']) #vp_a
            elif k == 2:
                st.append(df['f']) #extract f
            elif k == 4:
                st.append(df['ip_a_diff_grad']) #extract gradient of positive current angle column from each PMU 
            elif k == 5:
                st.append(df['f_grad']) #extract gradient of freq column from each PMU 
            # elif k == 0:
                # st.append(df['df']) #extract ROCOF column from each PMU 
                
                
            if(min>df['df'].shape[0]):
                min = df['df'].shape[0]
            #st = np.append(st, df['v_grad'].to_numpy(), axis=0)
        elif os.path.exists(filename) ==0:
           print('fail to read')
     
        # print('i = ',i)    
        # print(rootpath[i])
        if((i+1)%23==0):
            j = j+1
            print('j = ',j)
            st = np.array(st) 
            if (min==num_time_sample and flag ==1):
                features.append(st)
                
                
                event_name = rootpath[i]
                ename.append(event_name)
                planned_event=['Planned Operations','Planned Service', 'Planned Testing']
                #lABEL ASSIGNMENT
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
            st=[]       
            min=num_time_sample
            flag = 1
        
    print(j)
    features=np.array(features) 
    print(features.shape)
    num_samples,h,w=features.shape
    features=features.reshape(num_samples, h, w,1) # add a channel dimension in order for Keras to be compatible

    list = ['vp_m','vp_a','f']
    name = list[k]
    path3 = 'pickleset/'
    # save features for all the events
    pickle_out = open(path3 + "X2_S"+str(num)+"_"+str(name)+"_6.pickle","wb")
    pickle.dump(features, pickle_out, protocol=2)
    pickle_out.close() 
    
    b = np.array(ename)
    #save labels for all the events
    labels=np.array(labels)
    labels = labels.reshape(num_samples,-1)
    b = b.reshape(num_samples,-1)
    # print(labels.shape)
    # print(b.shape)
    labels=np.concatenate((b, labels), axis=1)
    
    print(labels.shape)
    pickle_out = open(path3 + "y2_S"+str(num)+"_"+str(name)+"_6.pickle","wb")
    pickle.dump(labels, pickle_out, protocol=2)
    pickle_out.close()  

'''
# save name of events that had missing values in their PMUs
pickle_out = open("missing_event.pickle","wb")
pickle.dump(missing_event, pickle_out, protocol=2)
pickle_out.close()   
'''
# for i in range(1,5+1):



def main(num):    
    s1 = timeit.default_timer() 
    for i in range(4,5+1):
        deal(i, num)
    # read()
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
if __name__ == '__main__':  


    num = int(sys.argv[1])
    

    main(num)





   


    
