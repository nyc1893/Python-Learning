# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:12:02 2020

@author: Hanif , Amir
"""

import sys, os
import pandas as pd
import numpy as np
import timeit
import pyarrow.parquet as pq
import math
import timeit
import pickle
from datetime import datetime, timedelta

from ED_OLAP import get_feature
from ED_OLAP import get_single_feature
# from scipy.io import loadmat
# from matrix_completion import *
from scipy.linalg import svd


mm = 9



def cc(dd,ii,feature):
    dir_loc= '/home/ycliu/DOE_test/2016B/theMonth='+str(mm)+'/theDay='+str(dd)+'/id='+list_60hz[ii]+'/' 
    dir_list=os.listdir(dir_loc)

    sum = 0

    print(len(dir_list))
    k = 0
    i = dir_list[k] 
    ip_loc=dir_loc+i # this line takes each day iteratively
    df = pd.read_parquet(ip_loc)
    v = df.shape[0]

    df2 = df[df['vp_m'].notna()]
    v1 = df2.shape[0]
    df3 = df2[df2['f'].notna()]    
    dt1 = df3
    # get the pure not NAN data
    for k in range(1,len(dir_list)):
        i = dir_list[k] 
        ip_loc=dir_loc+i # this line takes each day iteratively
        df = pd.read_parquet(ip_loc)
        v = df.shape[0]
        # print(v)
        df2 = df[df['vp_m'].notna()]
        # v1 = df2.shape[0]
        df3 = df2[df2['f'].notna()]
        dt2 = df3
        dt1 = pd.concat([dt1,dt2])
    dt1 = dt1.reset_index()
    dt1['time'] = pd.to_datetime(dt1['index'],format = '%Y-%m-%d %H:%M:%S.SSS')
    dt1.pop("index")
    dt1.sort_values('time', inplace=True)  
    dt1.set_index(["time"], inplace=True)   
    ll =["vp_m","vp_a","ip_m","ip_a","f","f"]
    if(dt1.shape[0]) == 5184000:
        return dt1[ll[feature]]
        
    k = 0
    i = dir_list[k] 
    ip_loc=dir_loc+i # this line takes each day iteratively
    df = pd.read_parquet(ip_loc)
    # get the time index 
    for k in range(1,len(dir_list)):
        i = dir_list[k] 
        ip_loc=dir_loc+i # this line takes each day iteratively
        df2 = pd.read_parquet(ip_loc)    
        df = pd.concat([df,df2])
    df = df.reset_index()
    df['time'] = pd.to_datetime(df['index'],format = '%Y-%m-%d %H:%M:%S.SSS')        
    df.pop("index")
    df.sort_values('time', inplace=True)   
    df.drop_duplicates(subset=['time'], keep='first', inplace=True)    
    df.set_index(["time"], inplace=True)  
    
    df["vp_m"] = dt1["vp_m"]
    df["vp_a"] = dt1["vp_a"]
    df["ip_m"] = dt1["ip_m"]
    df["ip_a"] = dt1["ip_a"]
    df["f"] = dt1["f"]
    

    # print(dt1.shape)
    # print(df.shape)
    # print(df.isna().sum())
    return df[ll[feature]]
    
def gen_oneday_feature(day,feature):  
    ll =["vpm","vpa","ipm","ipa","f","f"]
    for j in range(len(list_60hz)):
        dir_loc= '/home/ycliu/DOE_test/2016B/theMonth='+str(mm)+'/theDay='+str(day)+'/id='+list_60hz[j]+'/' # this directory should be updated based on your computer.
        if os.path.isdir(dir_loc):
            df1 = pd.DataFrame(cc(day,j,feature))
            df1.columns = [list_60hz[j]]
            break    
    
    for i in range(j+1,len(list_60hz)):
        print(list_60hz[i])
        
        dir_loc= '/home/ycliu/DOE_test/2016B/theMonth='+str(mm)+'/theDay='+str(day)+'/id='+list_60hz[i]+'/' # this directory should be updated based on your computer.
        if os.path.isdir(dir_loc):            
            df2 = pd.DataFrame(cc(day,i,feature))
            df1[list_60hz[i]] = df2

    

    print(df1.head())
    print(df1.shape)
    df1.to_parquet("oneday/m="+str(mm)+"_day="+str(day)+"_"+ll[feature]+".parquet")
    

    
def rd(data,k):

    data = data.T
    # print(data.shape)
    # print(data.head())

    st1 =[]

    
    for j in range(0,data.shape[0]):

        if(k == 0):
            temp = data[j,:]/np.mean(data[j,:])
        elif(k == 1):
            temp = np.deg2rad(data[j,:])
        elif(k == 2):
            temp = data[j,:]/100             
        elif(k == 3):
            temp = np.deg2rad(data[j,:])                
        elif(k == 4):
            temp = data[j,:]/np.mean(data[j,:])            
        elif(k == 5):
            temp = data[j,:]      
            # st2.append(temp)
        st1.append(temp)

    st1 = np.array(st1)
    st1 = st1[:,np.newaxis,:]
    print(st1.shape)
    # print(st1[:1])
    return st1
    


def filer_null(df):

    dt = df.dropna(axis = 1, how= 'any')

    return dt.values
    
def test(word):
    path = "un_list/"

    # df = pd.read_csv(path+word+".csv")

    df = pd.read_csv(word+".csv")
    df = df["name"].values
    path1 = "../vpm/"
    path2 = "../ipm/"
    path3 = "../freq/"
    path5 = "zeta/"
    # df.shape[0]
    res =[]
    for i in range(df.shape[0]):
        f1 = path1+df[i]+".parquet"
        f2 = path2+df[i]+".parquet"
        f3 = path3+df[i]+".parquet"
        if os.access(f1, os.R_OK) and os.access(f2, os.R_OK) and os.access(f3, os.R_OK):
            try:
                name = df[i]
                temp = pq.read_table(path1+ name+".parquet").to_pandas()
                temp = filer_null(temp)
                if(temp.shape[0]>5):
                    temp = temp.T
                    temp = temp[:,np.newaxis,:]
                    # print(temp.shape)
                    res = get_single_feature(temp,30)
                    print(res.shape)
                    
                    temp = pq.read_table(path2+ name+".parquet").to_pandas()
                    temp = filer_null(temp)
                    temp = temp.T
                    temp = temp[:,np.newaxis,:]         
                    res2 = get_single_feature(temp,30)
                    res = np.concatenate([res,res2])
                    
                    temp = pq.read_table(path3+ name+".parquet").to_pandas()
                    temp = filer_null(temp)
                    temp = temp.T
                    temp = temp[:,np.newaxis,:]         
                    res2 = get_single_feature(temp,30)
                    res = np.concatenate([res,res2])               
                    
                    pickle_out = open(path5 + name+".pickle","wb")
                    pickle.dump(res, pickle_out, protocol=2)
                    pickle_out.close()   
                    print(name+" save done")
            except:
                print("error")
def batch():
    for i in range(1,13):
        word = "y2017_m"+str(i)
        test(word)
            


def main():
    s1 = timeit.default_timer()
            
    batch()

    s2 = timeit.default_timer()
    print('Time:(min) ', (s2 - s1)/60 )
if __name__ == "__main__":
    main()
    


    
