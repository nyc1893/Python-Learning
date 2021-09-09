# This code is to find out the mutilabel events and it statics

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

import pickle
import os
import sys

from datetime import datetime

import timeit



def readfilelist(name):

    # name = 'y2016_m5'
    path = "./"+name+"/"
    path2 = "../../ic_c/" + name
    fileList=os.listdir(path2)
    fileList.sort(key= lambda x:str(x))
    sum = 0
    list2 = []
    for f in fileList:
        if(f.endswith(".parquet")):
            if("Line" in f or "Generator" in f or "Osc" in f):
                x = f.split("_")
                temp = x[0]+"_"+x[1]+"_"+x[2]+"_"+x[3]+"_"+x[4]
                list2.append(temp)
                sum+=1
    # print(list2)
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
    tt.to_csv(name+".csv",index = None)

def save_freq(name):

    # name = 'y2016_m5'
    path = "./"+name+"/"
    path2 = "../../ic_c/" + name
    df = pd.read_csv(name+".csv").values
    # print(df[0,1])
    # temp = pq.read_table(source=path2+"/2016_Jan_31_0_Generator_C993.parquet").to_pandas()
    # st = []
    # st.append(temp["f"].values)
    # print(st[:5])
    # print(temp.head())
    fileList=os.listdir(path2)

    v = 0
    for ii in range(df.shape[0]):
        print(df[ii,1])
        st1 =[]
        st2 =[]
        for f in fileList:
            if(f.endswith(".parquet") and v<999):
                if(df[ii,1] in f):
                    x = f.split("_")

                    temp = pq.read_table(source=path2+"/"+f).to_pandas()
                    # print(temp.shape)
                    # print(temp.head())
                    a = temp["f"].values
                    if(a.shape[0] == 18000):
                        st1.append(a)
                        st2.append(x[5].split(".")[0])
                        v+=1
        # st
        st1 = pd.DataFrame(st1)
        st1 = st1.transpose()
        # print(st1.shape)
        # print(len(st2))
        st1.columns = st2
        # print(st1.head())
        savepath = "../freq/"
        # st1.to_csv("f_"+df[0,1]+".csv",index= 0)
        st1.to_parquet(savepath+df[ii,1]+".parquet") 


def save_freq(name):

    # name = 'y2016_m5'
    path = "./"+name+"/"
    path2 = "../../ic_c/" + name
    df = pd.read_csv(name+".csv").values
    # print(df[0,1])
    # temp = pq.read_table(source=path2+"/2016_Jan_31_0_Generator_C993.parquet").to_pandas()
    # st = []
    # st.append(temp["f"].values)
    # print(st[:5])
    # print(temp.head())
    fileList=os.listdir(path2)

    v = 0
    for ii in range(df.shape[0]):
        print(df[ii,1])
        st1 =[]
        st2 =[]
        for f in fileList:
            if(f.endswith(".parquet") and v<999):
                if(df[ii,1] in f):
                    x = f.split("_")

                    temp = pq.read_table(source=path2+"/"+f).to_pandas()
                    # print(temp.shape)
                    # print(temp.head())
                    a = temp["f"].values
                    if(a.shape[0] == 18000):
                        st1.append(a)
                        st2.append(x[5].split(".")[0])
                        v+=1
        # st
        st1 = pd.DataFrame(st1)
        st1 = st1.transpose()
        # print(st1.shape)
        # print(len(st2))
        st1.columns = st2
        # print(st1.head())
        savepath = "../freq/"
        # st1.to_csv("f_"+df[0,1]+".csv",index= 0)
        st1.to_parquet(savepath+df[ii,1]+".parquet") 


def save_vpm(name):

    # name = 'y2016_m5'
    path = "./"+name+"/"
    path2 = "../../ic_c/" + name
    df = pd.read_csv(name+".csv").values
    # print(df[0,1])
    # temp = pq.read_table(source=path2+"/2016_Jan_31_0_Generator_C993.parquet").to_pandas()
    # st = []
    # st.append(temp["f"].values)
    # print(st[:5])
    # print(temp.head())
    fileList=os.listdir(path2)

    v = 0
    for ii in range(df.shape[0]):
        print(df[ii,1])
        st1 =[]
        st2 =[]
        for f in fileList:
            if(f.endswith(".parquet") and v<999):
                if(df[ii,1] in f):
                    x = f.split("_")

                    temp = pq.read_table(source=path2+"/"+f).to_pandas()
                    # print(temp.shape)
                    # print(temp.head())
                    a = temp["vp_m"].values
                    if(a.shape[0] == 18000):
                        st1.append(a)
                        st2.append(x[5].split(".")[0])
                        v+=1
        # st
        st1 = pd.DataFrame(st1)
        st1 = st1.transpose()
        # print(st1.shape)
        # print(len(st2))
        st1.columns = st2
        # print(st1.head())
        savepath = "../vpm/"
        # st1.to_csv("f_"+df[0,1]+".csv",index= 0)
        st1.to_parquet(savepath+df[ii,1]+".parquet") 

def save_ipm(name):

    # name = 'y2016_m5'
    path = "./"+name+"/"
    path2 = "../../ic_c/" + name
    df = pd.read_csv(name+".csv").values
    # print(df[0,1])
    # temp = pq.read_table(source=path2+"/2016_Jan_31_0_Generator_C993.parquet").to_pandas()
    # st = []
    # st.append(temp["f"].values)
    # print(st[:5])
    # print(temp.head())
    fileList=os.listdir(path2)

    v = 0
    for ii in range(df.shape[0]):
        print(df[ii,1])
        st1 =[]
        st2 =[]
        for f in fileList:
            if(f.endswith(".parquet") and v<999):
                if(df[ii,1] in f):
                    x = f.split("_")

                    temp = pq.read_table(source=path2+"/"+f).to_pandas()
                    # print(temp.shape)
                    # print(temp.head())
                    if (temp["ip_m"].values.shape[0]!=0):
                        a = temp["ip_m"].values
                        if(a.shape[0] == 18000):
                            st1.append(a)
                            st2.append(x[5].split(".")[0])
                            v+=1
        # st
        st1 = pd.DataFrame(st1)
        st1 = st1.transpose()
        # print(st1.shape)
        # print(len(st2))
        st1.columns = st2
        # print(st1.head())
        savepath = "../ipm/"
        # st1.to_csv("f_"+df[0,1]+".csv",index= 0)
        st1.to_parquet(savepath+df[ii,1]+".parquet") 

def read_event():
    path2 = "../../ic_c/y2016_m1" 
    temp = pq.read_table(source=path2+"/2016_Jan_30_0_Line_C930.parquet").to_pandas()
    temp.head(15).to_csv("ss.csv",index = 0)
    
def main():

    list3 = list(range(13))
    # kk = int(sys.argv[1])
    for kk in range(2,12+1):
        # readfilelist("y2016_m"+str(list3[kk]))
        save_vpm("y2016_m"+str(list3[kk]))
    
    # read_event()
if __name__ == '__main__':  

    main()
