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
from scipy.stats.distributions import chi2
from collections import Counter
from openpyxl import *
import xlsxwriter
from datetime import datetime, timedelta

#PMU_list = pd.read_excel('/home/hanif/PMU_reporting_rate_corrected.xlsx', sheet_name='Sheet1')
#PMU_list_30 = PMU_list['30fps']
PMU_list_30_limited = np.array(['608','740','785','505','209'])
#PMU_list_60 = PMU_list['60fps']

#n_PMU_30 = PMU_list_30.size - PMU_list_30.isna().sum() - 1
n_PMU_30_limited = len(PMU_list_30_limited)
#n_PMU_60 = PMU_list_60.size - PMU_list_60.isna().sum() - 1

start = timeit.default_timer()

# dir_loc= '/pnnl1/ic_b/theYear=2017/theMonth=3/theDay=1/'




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


mm =6



def cc(dd,ii):
    dir_loc= '/home/ycliu/DOE_test/2016B/theMonth='+str(mm)+'/theDay='+str(dd)+'/id='+list_60hz[ii]+'/' 
    dir_list=os.listdir(dir_loc)

    sum = 0
    # for i in dir_list:
    print(len(dir_list))
    for k in range(len(dir_list)):
        i = dir_list[k] 
        ip_loc=dir_loc+i # this line takes each day iteratively
        # print(i)

        df = pd.read_parquet(ip_loc)
        v = df.shape[0]
        # print(v)
        df2 = df[df['vp_m'].notna()]
        v1 = df2.shape[0]
        df3 = df2[df2['f'].notna()]
        v2 = df3.shape[0]
        # print(v2)
        df3 = df2[df2['ip_m'].notna()]
        df3 = df3.reset_index()
        dt = df3["index"].values


        res = []
        for j in range(df3.shape[0]):
            temp = str(dt[j]).split("T")
            x = temp[1].split(":")
            # print(temp[0] + " "+x[0]+ ":"+x[1])
            # res.append(temp[0] + " "+x[0])
            res.append(temp[0] + " "+x[0]+ ":"+x[1])
        import collections
        a = np.array(res)
        b = collections.Counter(a)
        res = []
        for item in b.items():
            # print(item)
            res.append(str(item))
        # print(len(res))
        sum = 0

        dff = pd.DataFrame(res)
        dff.to_csv("temp/see"+str(k)+".csv",index= 0)


    df =  pd.read_csv("temp/see0.csv")
    num = len(dir_list)
    for i in range(1,num):
        df2 = pd.read_csv("temp/see"+str(i)+".csv")
        df = pd.concat([df,df2])
    dt= df["0"].values
    st1= []
    st2 =[]
    for i in range(dt.shape[0]):
        x = dt[i].split("',")
        x1 = x[0].split("'")[1]
        x2 = x[1].split(")")[0]
        # print(x1)
        st1.append(x1)
        st2.append(x2)
        
    df["time"] = st1
    df["num"] = st2
    df.pop("0")
    df['time'] = pd.to_datetime(df['time'],format = '%Y-%m-%d %H:%M')


    df.sort_values('time', inplace=True)    
    df['num'] = pd.to_numeric(df.num)
    dt = df[df["num"]!=3600]
    dt["id"] =list_60hz[ii]
    df.to_csv("log/"+str(mm)+"_"+str(dd)+"_"+str(list_60hz[ii])+".csv",index= 0)
    return dt

def day(dd):
    for j in range(len(list_60hz)):

        dir_loc= '/home/ycliu/DOE_test/2016B/theMonth='+str(mm)+'/theDay='+str(dd)+'/id='+list_60hz[j]+'/' # this directory should be updated based on your computer.
        if os.path.isdir(dir_loc):
            df = cc(dd,j)
            break


            
    for i in range(j+1,len(list_60hz)):
        print(list_60hz[i])
        
        dir_loc= '/home/ycliu/DOE_test/2016B/theMonth='+str(mm)+'/theDay='+str(dd)+'/id='+list_60hz[i]+'/' # this directory should be updated based on your computer.
        if os.path.isdir(dir_loc):            
            df2 = cc(dd,i)

            df = pd.concat([df,df2])
        
    df.to_csv("log/oneday_"+str(mm)+"_"+str(dd)+"_"+".csv",index= 0)
    
def main():
    for i in range(17,32):
        day(i)
        
if __name__ == "__main__":
    main()
    
# for j in range(len(res)):
    # temp = res[j].split(",")[1].split(")")[0]
    # print(type(temp))
    # print(temp)
    # if(temp!=" 3600"):
        # print(res[j])
        
        # sum+=1
# print(sum)
    
# print(res[:10])
# c = pd.DataFrame(b)
# print(c.head())
# print(type(b))
# print(b.most_common(1))   
# print(b.most_common(2))
# print(df3.head(10))
# v3 = df3.shape[0]


# print(v3)
# sum+= v3
# print("sum=",sum)



    
