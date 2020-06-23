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
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
# import cv2
import timeit

start = timeit.default_timer()

# min-max normalize each of PMU mesurements ([0 1])
# def normalize(x):
#     x=( x-np.min(x) )/ ( np.max(x)-np.min(x) )
#     return x
    

#Constants
num_time_sample=36000
num_pmu=23

#loading event log
event_log = pd.read_csv('../../B_Training.csv')


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
for i in range(1,23*296+1):
    

    filename = path1 + rootpath[i-1]
    # print('i = ',i)
    if os.path.exists(filename) ==1:
        df=pq.read_table(filename).to_pandas()
        # print('hahas')
        
        #st.append(df['v_grad']) #extract gradient of voltage column from each PMU
        #st.append(df['i_grad']) #extract gradient of current column from each PMU
        #st.append(df['vp_a_diff_grad']) #extract gradient of positive voltage angle column from each PMU 
        #st.append(df['ip_a_diff_grad']) #extract gradient of positive current angle column from each PMU 
        # st.append(df['f_grad']) #extract gradient of freq column from each PMU 
        st.append(df['df']) #extract ROCOF column from each PMU 
        if(min>df['df'].shape[0]):
            min = df['df'].shape[0]
        #st = np.append(st, df['v_grad'].to_numpy(), axis=0)
    elif os.path.exists(filename) ==0:
       print('fail to read')
        

        
        
    event_name = rootpath[i-1]
    # planned_event=['Planned Operations','Planned Service', 'Planned Testing']
    #lABEL ASSIGNMENT
    if 'Line' in event_name:
        # temp=event_log['Cause'][iteration]
        # if temp==planned_event[0] or temp==planned_event[1] or temp==planned_event[2]:
           # labels.append(4)
        # else:
        labels.append(0)
            
    elif 'XFMR' in event_name:
        # temp=event_log['Cause'][iteration]
        # if temp==planned_event[0] or temp==planned_event[1] or temp==planned_event[2]:
           # labels.append(5)
        # else:
        labels.append(1)
              
    elif 'Frequency' in event_name: 
        labels.append(2)
    elif 'Oscillation' in event_name: 
        labels.append(3)

        #Convert the matrix to image and save
        #cv2.imwrite('I:/IBM data/Events/B/Images/8 weeks-1/Ip_angle_diff_gradient/Training/'+str(event_name)+'.jpg', st)

        
    if((i)%23==0 and i!=1):
    j = j+1
    print('j = ',j)
    st = np.array(st) 
    if min==num_time_sample:
        features.append(st)
        
    st=[]       
    min=num_time_sample

print(j)
features=np.array(features) 
print(features.shape)
num_samples,h,w=features.shape
features=features.reshape(num_samples, h, w,1) # add a channel dimension in order for Keras to be compatible

# save features for all the events
pickle_out = open("X_train_6.pickle","wb")
pickle.dump(features, pickle_out, protocol=2)
pickle_out.close() 

# save labels for all the events
labels=np.array(labels) 
pickle_out = open("y_train_6.pickle","wb")
pickle.dump(labels, pickle_out, protocol=2)
pickle_out.close()  

'''
# save name of events that had missing values in their PMUs
pickle_out = open("missing_event.pickle","wb")
pickle.dump(missing_event, pickle_out, protocol=2)
pickle_out.close()   
'''
  
stop = timeit.default_timer()
#running time
print('Time: ', stop - start)

         





   


    
