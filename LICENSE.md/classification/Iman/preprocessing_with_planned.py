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
path1 = '../8weeks_2/'
path2 = '../cc/'
# with os.scandir('/Volumes/Iman/IBM data/Events/B/Original/8 weeks-1/') as entries:
with os.scandir(path1) as entries:
    for entry in entries:
        iteration+=1
        event_name_full=entry.name
        event_name=event_name_full.split(".")[0]
        print(event_name)
        

            
        
        #FEATURES ASSIGNMENT
        #spatiotemporal matrix of location and time of event
        st=[]
               
        for pmu_id_index in range(PMU_list_1.size):
            current_id=PMU_list_1[pmu_id_index]
            ## loading datasets
            filename = path2 +str(event_name)+'_'+str(current_id)+'.parquet'
            if os.path.exists(filename) ==1:
            
      
                df=pq.read_table(filename).to_pandas()
                
                
                #st.append(df['v_grad']) #extract gradient of voltage column from each PMU
                #st.append(df['i_grad']) #extract gradient of current column from each PMU
                #st.append(df['vp_a_diff_grad']) #extract gradient of positive voltage angle column from each PMU 
                #st.append(df['ip_a_diff_grad']) #extract gradient of positive current angle column from each PMU 
                # st.append(df['f_grad']) #extract gradient of freq column from each PMU 
                st.append(df['df']) #extract ROCOF column from each PMU 
                #st = np.append(st, df['v_grad'].to_numpy(), axis=0)
        
        #Convert series to numpy  
        st = np.array(st) 
        
        #figure out size of each matrix and if time samples are not equal to the specified value they will be excluded from dataset
        # Or if there is nan in any of values of st don't include that event
        size=st.shape
        # if (size[1]!=num_time_sample) or(np.isnan(np.sum(st))):
            # count=count+1
            # print(count)
            # missing_event.append(str(event_name)) 
            # continue

        
        features.append(st)
        
        planned_event=['Planned Operations','Planned Service', 'Planned Testing']
        #lABEL ASSIGNMENT
        if 'Line Outage' in event_name:
            temp=event_log['Cause'][iteration]
            if temp==planned_event[0] or temp==planned_event[1] or temp==planned_event[2]:
               labels.append(4)
            else:
                labels.append(0)
                
        elif 'XFMR Outage' in event_name:
            temp=event_log['Cause'][iteration]
            if temp==planned_event[0] or temp==planned_event[1] or temp==planned_event[2]:
               labels.append(5)
            else:
                labels.append(1)
                  
        elif 'Frequency Event' in event_name: 
            labels.append(2)
        elif 'Oscillation Event' in event_name: 
            labels.append(3)

        #Convert the matrix to image and save
        #cv2.imwrite('I:/IBM data/Events/B/Images/8 weeks-1/Ip_angle_diff_gradient/Training/'+str(event_name)+'.jpg', st)

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

         





   


    
