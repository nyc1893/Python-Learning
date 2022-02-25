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
from scipy.io import loadmat
# from matrix_completion import *
from scipy.linalg import svd

list_60hz =[
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
'B992'
]


mm = 9



def cc(dd,ii,feature):
    dir_loc= '/home/ycliu/DOE_test/2016B/theMonth='+str(mm)+'/theDay='+str(dd)+'/id='+list_60hz[ii]+'/' 
    dir_list=os.listdir(dir_loc)

    sum = 0
    # for i in dir_list:
    print(len(dir_list))
    k = 0
    i = dir_list[k] 
    ip_loc=dir_loc+i # this line takes each day iteratively
    df = pd.read_parquet(ip_loc)
    v = df.shape[0]
    # print(v)
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
    
def gen_oneday_ROCOF(m,day):  
    df = pd.read_parquet("oneday/m="+str(m)+"_day="+str(day)+"_f.parquet")
    # df = df.iloc[0:10]
    ly = df.columns.values.tolist()    

    for i in ly:
        df[i+"_1"] = df[i].shift(-1)
        df[i] = df[i] - df[i+"_1"]
    df= df[ly]
    df.dropna(axis=0,how='all')  
    
    df.to_parquet("oneday/m="+str(m)+"_day="+str(day)+"_rof.parquet")
    print(df.head())
    print("oneday/m="+str(m)+"_day="+str(day)+"_save done")
    
def rd(data,k):

    data = data.T
    print(data.shape)
    # print(data.dtype)
    data = data.astype(np.float64)
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
    
def datapack(m,day,hour):
    # try:
    # data = rd_recover(m,day,hour,0)
    data = filer_null(m,day,hour,0)
    # data.astype(np.float64)
    print("aa ",data.shape)
    if(data.shape[1]!=0):
        df1 = rd(data,0)
    
        for i in range(1,6):
            # data = rd_recover(m,day,hour,i)
            data = filer_null(m,day,hour,i)
            # data.astype(np.float64)
            # filer_null
            if(data.shape[1]!=0):
                df2 = rd(data,i)
                df1 = np.concatenate((df1, df2), axis=1) 
            
        path3 = 'oneday17/zeta/'
        # save features for all the events
        print(df1.shape)
        # if(df1.shape[0]<23):
        X_train = get_feature(df1,30)
        pickle_out = open(path3 + "m="+str(m)+"_day="+str(day)+"_h="+str(hour)+".pickle","wb")
        pickle.dump(X_train, pickle_out, protocol=2)
        pickle_out.close()   
        print("save done")

    # except:
        # print("has error")





def get_rof():

    # m = 2
    # for day in range(13,25+1):
        # gen_oneday_ROCOF(m,day)    
    # m = 4
    # for day in range(8,21+1):
        # gen_oneday_ROCOF(m,day)
        
  
    m = 6
    l_day =[3,6,7,8]
    for day in l_day:
        gen_oneday_ROCOF(m,day) 
    for day in range(11,16+1):
        gen_oneday_ROCOF(m,day)  

    # m = 7
    # for day in range(29,31+1):
        # gen_oneday_ROCOF(m,day)  
        
    # m = 8
    # l_day =[1,3,4]
    # for day in range(6,11+1):
        # gen_oneday_ROCOF(m,day)      
    # for day in range(6,11+1):
        # gen_oneday_ROCOF(m,day)  
        
    m = 9
    for day in range(23,30+1):
        gen_oneday_ROCOF(m,day)  
        
def filer_null(m,day,hour,feature):

    # ll =["vpm","vpa","ipm","ipa","f","f"]
    ll = ["vp_m","vp_a","ip_m","ip_a","f","f"]
    
    # df = pd.read_parquet("oneday17/ex_m="+str(m)+"_day="+str(day)+"_"+ll[feature]+".parquet")
    
    df = pd.read_parquet("oneday17/m="+str(m)+"_day="+str(day)+"_"+ll[feature]+".parquet")
    # print(df.dtypes)
    # data.astype('float')
    dt = df.iloc[10800*20*(hour-1):10800*20*(hour)]
    # if(dt.isnull().sum().sum()==0):
        # print("clear")
    # else:
        # print(hour)
        # print(dt.isnull().sum())
    # print(dt.head())
    # print(dt.shape)
    
    dt.replace("", np.nan, inplace=True)
    dt = dt.dropna(axis = 1, how= 'any')
    # print(dt.shape)
    # print(dt.isnull().sum())
    return dt.values
    


    
def test(m,day,hour):
    path = "oneday17/"
    name = "m="+str(m)+"_day="+str(day)+"_"
    f1 = path+name+"vp_m.parquet"
    f2 = path+name+"ip_m.parquet"
    f3 = path+name+"f.parquet"
    
    path5 = "oneday17/zeta/"
    if os.access(f1, os.R_OK) and os.access(f2, os.R_OK) and os.access(f3, os.R_OK):
        try:

            temp = pq.read_table(f1).to_pandas()
            temp = filer_null(m,day,hour,0)
            if(temp.shape[0]>5):
                temp = temp.T
                temp = temp[:,np.newaxis,:]
                # print(temp.shape)
                res = get_single_feature(temp,30)
                print(res.shape)
                
                temp = pq.read_table(f2).to_pandas()
                temp = filer_null(m,day,hour,2)
                temp = temp.T
                temp = temp[:,np.newaxis,:]         
                res2 = get_single_feature(temp,30)
                res = np.concatenate([res,res2])
                
                temp = pq.read_table(f3).to_pandas()
                temp = filer_null(m,day,hour,4)
                temp = temp.T
                temp = temp[:,np.newaxis,:]         
                res2 = get_single_feature(temp,30)
                res = np.concatenate([res,res2])               
                
                pickle_out = open(path5 + name+"h="+str(hour)+".pickle","wb")
                pickle.dump(res, pickle_out, protocol=2)
                pickle_out.close()   
                print(name+"h="+str(hour)+" save done")
        except:
            print("error")
        
def batch():
    m = 1
    for day in range(20,25+1):
        for hour in range(1,24+1):
        
            test(m,day,hour)
            
def main():
    s1 = timeit.default_timer()

    batch()

    
    s2 = timeit.default_timer()
    print('Time:(min) ', (s2 - s1)/60 )
if __name__ == "__main__":
    main()
    


    