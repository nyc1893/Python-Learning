# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:47:16 2020

@author: iniazazari

This code converts the string fields to float, interpolates the missing entries,
calculates the gradients of indices, and save the results for each event and each PMU
"""
## importing libariries
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import math


##unwrapping func
def unwrap_angle(df1):
    #input df1 is a dataframe
    df=df1.copy()

    
    df=np.radians(df)
    x=df.sort_index()
    array1=x.iloc[:len(df)-1]
    array2=x.iloc[1:]    
    
    diff=np.array(array2)-np.array(array1)
    idx=np.where(abs(diff)>=np.pi)[0]
    
    for i in idx:
        #print(i)
        x[i+1:]=x[i+1:]+np.pi*2*-1*np.sign(diff[i])
    
    return(x)


## list for missing PMUs for all events
all_missing_pmu_list=[]
path  = '../../8weeks_3/'
# with os.scandir('/Volumes/Iman/IBM data/Events/B/Original/8 weeks-1/') as entries:
with os.scandir(path) as entries:
    for entry in entries:
        event_name_full=entry.name
        event_name=event_name_full.split(".")[0]
      
    
    ## loading datasets
        filename= path +str(event_name_full)
    
        df=pq.read_table(filename).to_pandas()
        
        PMU_list = pd.read_csv('PMU_reporting_rate.csv')
        
        
        PMU_list_1 = PMU_list['60fps']
        ## list for missing PMUs for each event
        missing_pmu_list=[]
        
        
        ## refernce PMU calculations
        df.columns = df.columns.astype(str)
        
        for ref_id in range(PMU_list_1.size):
            Ref_PMU=PMU_list_1[ref_id] 
            flag1= Ref_PMU in df['id'].unique()
            if flag1 == False:
                continue
            else:
                break
             
        
        
        subset_ref = df[df['id'] == Ref_PMU]
        subset_ref=subset_ref.reset_index()
        subset_ref.replace('',np.nan,inplace=True)
        subset_ref.iloc[:,2:20]=subset_ref.iloc[:,2:20].astype(float)
        subset_ref.interpolate(inplace=True)
        
        vp_a_ref=subset_ref['vp_a'].reset_index()
        vp_a_ref.drop('index',axis=1,inplace=True)
        
        ip_a_ref=subset_ref['ip_a'].reset_index()
        ip_a_ref.drop('index',axis=1,inplace=True)
        
        
        missing_pmu_list.append(event_name)
        for pmu_id_index in range(PMU_list_1.size):
            
            
            ##identifying missing PMUs
            current_id=PMU_list_1[pmu_id_index]
            isPMUexist= current_id in df['id'].unique()
            flag2= not isPMUexist and type(current_id) is str
            if flag2:
                missing_pmu_list.append(current_id)
                continue
               
            ##filling out missing data    
            subset = df[df['id'] == current_id]
            subset=subset.reset_index()
            subset.columns = subset.columns.astype(str)
            subset.replace('',np.nan,inplace=True)
            subset.iloc[:,2:20]=subset.iloc[:,2:20].astype(float)
            subset.interpolate(inplace=True)
            
            vp_a_temp=subset['vp_a'].reset_index()
            vp_a_temp.drop('index',axis=1,inplace=True)
        
            ip_a_temp=subset['ip_a'].reset_index()
            ip_a_temp.drop('index',axis=1,inplace=True)
            
            ##calculating secondary indices
            d_v = np.gradient(subset.reset_index()['vp_m'])
            d_i = np.gradient(subset.reset_index()['ip_m'])
            d_f = np.gradient(subset.reset_index()['f'])
            
            vp_a_diff=unwrap_angle(vp_a_temp)-unwrap_angle(vp_a_ref)
            d_vp_a_diff = np.gradient(vp_a_diff['vp_a'])
            
            ip_a_diff=unwrap_angle(ip_a_temp)-unwrap_angle(ip_a_ref)
            d_ip_a_diff = np.gradient(ip_a_diff['ip_a'])
        
            
            ##Adding secondary indices to the dataframe
            subset['v_grad'] = d_v
            subset['i_grad'] = d_i
            subset['f_grad'] = d_f
            subset['vp_a_diff_grad']=d_vp_a_diff
            subset['ip_a_diff_grad']=d_ip_a_diff
            
            ##saving the data
            subset.columns = subset.columns.astype(str)
            save_file_name=str(event_name)+'_'+str(current_id)+'.parquet'
            subset.to_parquet(save_file_name)
            
            '''
            ##loading parquet file 
            loaded_df=pq.read_table('I:/IBM data/Code/2016_Jan_01_0_Line Outage_B126.parquet').to_pandas()
            '''
    
        ## save  missing PMUs for each event
        joined_missing_pmu_list = "_"
        # joins elements of list by '-' 
        # and stores in sting s 
        joined_missing_pmu_list = joined_missing_pmu_list.join(missing_pmu_list)
        all_missing_pmu_list.append(joined_missing_pmu_list)
    

## save  missing PMUs for all events
with open('all_missing_pmu_list', 'wb') as f:
    pickle.dump(all_missing_pmu_list, f)


'''
## read missing PMUs for all events    
with open('all_missing_pmu_list', 'rb') as f:   
    all_missing_pmu_list = pickle.load(f)
print(all_missing_pmu_list)    
'''





   


    
