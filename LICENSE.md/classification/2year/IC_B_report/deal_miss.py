# -*- coding: utf-8 -*-
"""
Created on 9/15/2021  
This one extract and package the Freq, vpm ipm data for zeta calculation
Also handle the some event that missing in precious Iman extracted version.
 
 save_freq2 is for the missing one,  save_freq is for the already interpolated data
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
import timeit
import ED_OLAP
import pickle as pickle 

def readfilelist(ii):

    # name = 'y2016_m5'

    path2 = "../8weeks_"+str(ii)+"_inter"
    fileList=os.listdir(path2)
    fileList.sort(key= lambda x:str(x))
    sum = 0
    list2 = []
    for f in fileList:
        if(f.endswith(".parquet")):
                x = f.split("_")
                temp = x[0]+"_"+x[1]+"_"+x[2]+"_"+x[3]+"_"+x[4]
                list2.append(temp)
                sum+=1
    st = pd.DataFrame(list2)
    st.columns = ['name']
    # df = st.groupby(["name"]).count().reset_index()
    # df.columns = ['name']
    # print(df.shape)
    
    tt= st["name"].value_counts()
    # print(type(tt))
    tt = pd.DataFrame(tt)
    tt.columns = ['count']
    tt["name"] = tt.index
    # print(type(tt))
    # for i in range(tt.shape[0]):
        # print(tt[i])
    # df = pd.DataFrame(tt)
    # print(tt)
    # print(df.shape)
    print(tt.shape)
    tt.to_csv("S_"+str(ii)+".csv",index = None)

def save_freq(ii):

    path2 = "../8weeks_"+str(ii)+"_inter"
    # fileList=os.listdir(path2)
    df = pd.read_csv("S_"+str(ii)+".csv").values
    # print(df[0,1])
    # temp = pq.read_table(source=path2+"/2016_Jan_31_0_Generator_C993.parquet").to_pandas()
    # st = []
    # st.append(temp["f"].values)
    # print(st[:5])
    # print(temp.head())
    fileList=os.listdir(path2)


    for j in range(df.shape[0]):
        print(df[j,1])
        st1 =[]
        st2 =[]
        for f in fileList:
            if(f.endswith(".parquet")):
                if(df[j,1] in f):
                    x = f.split("_")

                    temp = pq.read_table(source=path2+"/"+f).to_pandas()
                    # print(temp.shape)
                    # print(temp.head())
                    a = temp["f"].values
                    if(a.shape[0] == 36000):
                        st1.append(a)
                        st2.append(x[5].split(".")[0])

        # st
        st1 = pd.DataFrame(st1)
        st1 = st1.transpose()
        # print(st1.shape)
        # print(len(st2))
        st1.columns = st2
        ll = list(st1)
       
        for i in range(len(ll)):
            st1[ll[i]] = pd.to_numeric(st1[ll[i]])  
        # print(st1.head())
        savepath = "../freq/S_"+str(ii)+"/"
        # st1.to_csv("f_"+df[0,1]+".csv",index= 0)
        if(st1.shape[0]!=0):
            st1.to_parquet(savepath+df[j,1]+".parquet") 

def save_vpm(ii):

    path2 = "../8weeks_"+str(ii)+"_inter"
    # fileList=os.listdir(path2)
    df = pd.read_csv("S_"+str(ii)+".csv").values
    # print(df[0,1])
    # temp = pq.read_table(source=path2+"/2016_Jan_31_0_Generator_C993.parquet").to_pandas()
    # st = []
    # st.append(temp["f"].values)
    # print(st[:5])
    # print(temp.head())
    fileList=os.listdir(path2)


    for j in range(df.shape[0]):
        print(df[j,1])
        st1 =[]
        st2 =[]
        for f in fileList:
            if(f.endswith(".parquet")):
                if(df[j,1] in f):
                    x = f.split("_")

                    temp = pq.read_table(source=path2+"/"+f).to_pandas()
                    # print(temp.shape)
                    # print(temp.head())
                    a = temp["vp_m"].values
                    if(a.shape[0] == 36000):
                        st1.append(a)
                        st2.append(x[5].split(".")[0])

        # st
        st1 = pd.DataFrame(st1)
        st1 = st1.transpose()
        # print(st1.shape)
        # print(len(st2))
        st1.columns = st2
        ll = list(st1)
       
        for i in range(len(ll)):
            st1[ll[i]] = pd.to_numeric(st1[ll[i]])  
        # print(st1.head())
        savepath = "../vpm/S_"+str(ii)+"/"
        # st1.to_csv("f_"+df[0,1]+".csv",index= 0)
        if(st1.shape[0]!=0):
            st1.to_parquet(savepath+df[j,1]+".parquet") 


def save_ipm(ii):

    path2 = "../8weeks_"+str(ii)+"_inter"
    # fileList=os.listdir(path2)
    df = pd.read_csv("res/miss.csv").values
    # f_name = df[0,11]
    # print(df[0,1])
    # temp = pq.read_table(source=path2+"/2016_Jan_31_0_Generator_C993.parquet").to_pandas()
    # st = []
    # st.append(temp["f"].values)
    # print(st[:5])
    # print(temp.head())
    fileList=os.listdir(path2)


    for j in range(df.shape[0]):
        print(df[j,11])
        st1 =[]
        st2 =[]
        for f in fileList:
            if(f.endswith(".parquet") and df[j,11] in f):

                x = f.split("_")
                temp = pq.read_table(source=path2+"/"+f).to_pandas()
                a = temp["ip_m"].values
                # if(a.shape[0] == 36000):
                st1.append(a)
                st2.append(x[5].split(".")[0])

        # st
        st1 = pd.DataFrame(st1)
        st1 = st1.transpose()
        # print(st1.shape)
        # print(len(st2))
        st1.columns = st2
        ll = list(st1)
       
        for i in range(len(ll)):
            st1[ll[i]] = pd.to_numeric(st1[ll[i]])  
        # print(st1.head())
        savepath = "../ipm/miss/"
        # st1.to_csv("f_"+df[0,1]+".csv",index= 0)
        if(st1.shape[0]!=0):
            st1.to_parquet(savepath+df[j,11]+".parquet") 

def check_consistant():
    path1 = "../vpm/miss/"
    path2 = "../ipm/miss/"
    path3 = "../freq/miss/"
    fileList=os.listdir(path2)
    ll = []
    for f in fileList:
        if(f.endswith(".parquet")):
            temp1 = pq.read_table(path1+f).to_pandas()
            if(os.path.exists(path2+f) ==1 and  os.path.exists(path3+f) ==1):
                temp2 = pq.read_table(path2+f).to_pandas()
                temp3 = pq.read_table(path3+f).to_pandas()
                if(temp1.shape == temp2.shape and temp1.shape == temp3.shape ):
                    # print(f)
                    continue
                else:
                    ll.append(f)
    print(ll)

def proces(data,k):
    st1 =[]
    # data = np.squeeze(data)
    # print(data.shape)
    for j in range(0,data.shape[0]):


        if(k == 0):
            temp = data[j,:]/np.mean(data[j,:])

        elif(k == 1):
            temp = data[j,:]/100             
        elif(k == 2):

            temp = data[j,:]/np.mean(data[j,:])            

            # st2.append(temp)
        st1.append(temp)

    st1 = np.array(st1)
    st1 = st1[:,np.newaxis,:]

    return st1

def cal_zeta():
    path1 = "../vpm/miss/"
    path2 = "../ipm/miss/"
    path3 = "../freq/miss/"
    fileList=os.listdir(path2)
    ll = []
    for f in fileList:
        if(f.endswith(".parquet")):
            temp1 = pq.read_table(path1+f).to_pandas()
            temp1 = temp1.transpose().values
            temp1 = proces(temp1,0)
            # print(temp1.shape)
            
            temp2 = pq.read_table(path2+f).to_pandas()
            temp2 = temp2.transpose().values
            temp2 = proces(temp2,1)           

            temp3 = pq.read_table(path3+f).to_pandas()
            temp3 = temp3.transpose().values
            temp3 = proces(temp3,2)    
            
            temp1 = np.concatenate((temp1, temp2), axis=1)
            temp1 = np.concatenate((temp1, temp3), axis=1)
            # temp2 = pq.read_table(path2+f).to_pandas()
            temp1 = ED_OLAP.get_single_feature(temp1,30)
            
            path4 = "../zeta/miss/"
            # save features for all the events
            x= f.split(".")
            
            pickle_out = open(path4 + x[0]+".pickle","wb")
            pickle.dump(temp1, pickle_out, protocol=2)
            pickle_out.close()    
                
            print(temp1.shape)

def cal_zeta2():
    path1 = "../vpm/miss/"
    path2 = "../ipm/miss/"
    path3 = "../freq/miss/"
    fileList=os.listdir(path3)
    df = pd.read_csv("res/miss.csv")
    df =df[df["Category"] =="Frequency"]
    df = df["new"].values
    print(df.shape)
    for j in range(df.shape[0]):
        ll = []
        for f in fileList:
            if(f.endswith(".parquet") and df[j] in f):
                print(f)
                temp3 = pq.read_table(path3+f).to_pandas()
                temp3 = temp3.transpose().values
                temp3 = proces(temp3,2)    
                

                temp1 = ED_OLAP.get_single_feature(temp3,150)
                
                path4 = "../zeta_f/miss/"
                # save features for all the events
                x= f.split(".")
                pickle_out = open(path4 + x[0]+".pickle","wb")

                pickle.dump(temp1, pickle_out, protocol=2)
                pickle_out.close()    
                    
                print(temp1.shape)

import os
 
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print ("---  new folder...  ---")
		print ("---  OK  ---")
 
	else:
		print ("---  There is this folder!  ---")
		

def creat_folder():
    for i in range(1,14):
        file = "../zeta_f/S_"+str(i)
        mkdir(file) 
    
    

def check_complete2():

    path2 = "/scratch/share/extracted_data/8weeks_"+str(2)

    df = pd.read_csv("res/miss.csv").values
    path3 = "../../../"
    PMU_list = pd.read_csv(path3 + 'PMU_reporting_rate.csv')
    PMU_list_1 = PMU_list['60fps']
    fileList=os.listdir(path2)
    v =0 
    for i  in range(20):
        f_name = df[4,11]
        for f in fileList:
            if(f.endswith(".parquet") and f_name in f and v<1):
                
                x = f.split("_")
                temp = pq.read_table(path2+"/"+f).to_pandas()
                # temp.head(10).to_csv("see.csv")
                # temp['id'] = temp['id'].apply(lambda _: str(_))
                # print(temp["id"].dtype)
                for j in range(len(PMU_list_1)):
                    dt = temp[temp["id"] == PMU_list_1[j]]
                    # dt["f"]

                    print(j, dt["f"].shape,dt["ip_m"].shape,dt["vp_m"].shape)   
                v+=1
                
                
def save_ipm2(ii):

    path2 = "/scratch/share/extracted_data/8weeks_"+str(ii)

    df = pd.read_csv("res/miss.csv").values
    path3 = "../../../"
    PMU_list = pd.read_csv(path3 + 'PMU_reporting_rate.csv')
    PMU_list_1 = PMU_list['60fps']
    fileList=os.listdir(path2)

    # df.shape[0]
    for jj in range(df.shape[0]):
        print(df[jj,11])
        st1 =[]
        st2 =[]
        for f in fileList:
            if(f.endswith(".parquet") and df[jj,11]+"_"  in f):

                # x = f.split("_")
                print(f)
                temp = pq.read_table(source=path2+"/"+f).to_pandas()
                for j in range(len(PMU_list_1)):
                    dt = temp[temp["id"] == PMU_list_1[j]]
                    if(dt.shape[0]>0):
                        # print(dt.shape)
                        a = dt["ip_m"].values
                        st1.append(a)
                        st2.append(PMU_list_1[j])
                    
                st1 = pd.DataFrame(st1)
                st1 = st1.transpose()
                print(st1.shape)
                print(len(st2))
                st1.columns = st2
                ll = list(st1)
               
                for i in range(len(ll)):
                    st1[ll[i]] = pd.to_numeric(st1[ll[i]])  
                # print(st1.head())
                savepath = "../ipm/miss/"
                # st1.to_csv("f_"+df[0,1]+".csv",index= 0)
                if(st1.shape[0]!=0):
                    st1.to_parquet(savepath+df[jj,11]+".parquet") 

def save_vpm2(ii):

    path2 = "/scratch/share/extracted_data/8weeks_"+str(ii)

    df = pd.read_csv("res/miss.csv").values
    path3 = "../../../"
    PMU_list = pd.read_csv(path3 + 'PMU_reporting_rate.csv')
    PMU_list_1 = PMU_list['60fps']
    fileList=os.listdir(path2)

    # df.shape[0]
    for jj in range(df.shape[0]):
        print(df[jj,11])
        st1 =[]
        st2 =[]
        for f in fileList:
            if(f.endswith(".parquet") and df[jj,11]+"_" in f):

                # x = f.split("_")
                print(f)
                temp = pq.read_table(source=path2+"/"+f).to_pandas()
                for j in range(len(PMU_list_1)):
                    dt = temp[temp["id"] == PMU_list_1[j]]
                    if(dt.shape[0]>0):
                        # print(dt.shape)
                        a = dt["vp_m"].values
                        st1.append(a)
                        st2.append(PMU_list_1[j])
                    
                st1 = pd.DataFrame(st1)
                st1 = st1.transpose()
                print(st1.shape)
                print(len(st2))
                st1.columns = st2
                ll = list(st1)
               
                for i in range(len(ll)):
                    st1[ll[i]] = pd.to_numeric(st1[ll[i]])  
                # print(st1.head())
                savepath = "../vpm/miss/"
                # st1.to_csv("f_"+df[0,1]+".csv",index= 0)
                if(st1.shape[0]!=0):
                    st1.to_parquet(savepath+df[jj,11]+".parquet") 
def save_freq2(ii):

    path2 = "/scratch/share/extracted_data/8weeks_"+str(ii)

    df = pd.read_csv("res/miss.csv").values
    path3 = "../../../"
    PMU_list = pd.read_csv(path3 + 'PMU_reporting_rate.csv')
    PMU_list_1 = PMU_list['60fps']
    fileList=os.listdir(path2)

    # df.shape[0]
    for jj in range(df.shape[0]):
        print(df[jj,11])
        st1 =[]
        st2 =[]
        for f in fileList:
            if(f.endswith(".parquet") and df[jj,11]+"_"  in f):

                # x = f.split("_")
                print(f)
                temp = pq.read_table(source=path2+"/"+f).to_pandas()
                for j in range(len(PMU_list_1)):
                    dt = temp[temp["id"] == PMU_list_1[j]]
                    if(dt.shape[0]>0):
                        # print(dt.shape)
                        a = dt["f"].values
                        st1.append(a)
                        st2.append(PMU_list_1[j])
                    
                st1 = pd.DataFrame(st1)
                st1 = st1.transpose()
                print(st1.shape)
                print(len(st2))
                st1.columns = st2
                ll = list(st1)
               
                for i in range(len(ll)):
                    st1[ll[i]] = pd.to_numeric(st1[ll[i]])  
                # print(st1.head())
                savepath = "../freq/miss/"
                # st1.to_csv("f_"+df[0,1]+".csv",index= 0)
                if(st1.shape[0]!=0):
                    st1.to_parquet(savepath+df[jj,11]+".parquet") 
def main():
    s1 = timeit.default_timer()
    cal_zeta2()
    # check_consistant()
    # for ii in range(1,14):
        # save_ipm2(ii)
        # save_vpm2(ii)
        # save_freq2(ii)
    # check_complete2()
    s2 = timeit.default_timer()

    print('Time: ', (s2 - s1)/60 )        
if __name__ == "__main__":
    main()
