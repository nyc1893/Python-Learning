# This code is to find out the mutilabel events and it statics

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

import pickle
import os
import sys

from datetime import datetime

import timeit


def find_all_pmu():
    res=set([])

    for j in range(1,13):
        name2 = 'y2016_m'+str(j)
        path = "./"+name2+"/"
        path2 = "./" + name2
        rootpath =os.listdir(path2)
        
        rootpath.sort(key= lambda x:str(x))
        nn = len(rootpath)
        
        for i in range(0,nn):
            filename = path + rootpath[i]
            if os.path.exists(filename) ==1 and ".parquet" in rootpath[i]:
                temp = rootpath[i].split("C")
                temp2 = temp[1].split(".")
                res.add("C"+temp2[0])


    for j in range(1,13):
        name2 = 'y2017_m'+str(j)
        path = "./"+name2+"/"
        path2 = "./" + name2
        rootpath =os.listdir(path2)
        
        rootpath.sort(key= lambda x:str(x))
        nn = len(rootpath)
        
        for i in range(0,nn):
            filename = path + rootpath[i]
            if os.path.exists(filename) ==1 and ".parquet" in rootpath[i]:
                temp = rootpath[i].split("C")
                temp2 = temp[1].split(".")
                res.add("C"+temp2[0])
    print(res)
    print(len(res))
    res = pd.DataFrame(res)
    res.to_csv("cc.csv",index =None)

def read_all():
    
    fname ="stat-y2017_m"+str(1)+".csv"
    df = pd.read_csv(fname)
    for i in range(2,13):
        fname ="stat-y2017_m"+str(i)+".csv"
        df2 = pd.read_csv(fname)
        df = pd.concat([df,df2])
    print(df.shape)

def read2():
    path = "y2017_m12/"
    fname = "stat-y2017_m12.csv"
    df1 = pd.read_csv(fname).values
    df2 = pd.read_csv("cc.csv").values

    for j in range(df1.shape[0]):
        v = 0
        v2 = 0
        print(df1[j,0])
        for i in range(df2.shape[0]):
            filename = path+df1[j,0]+"_"+df1[j,2]+"_"+df2[i,0]+".parquet"
            v+=1
            if os.path.exists(filename) ==1 :
                v2+=1
                

        print(v,v2)

def save_f(mm):
    path = "y2016_m"+str(mm)+"/"
    fname = "stat-y2016_m"+str(mm)+".csv"
    df1 = pd.read_csv(fname).values
    df2 = pd.read_csv("cc.csv").values

    for j in range(df1.shape[0]):
        print(df1[j,0])
        v = 0
        v2 = 0        
        st = []
        try:
            for k in range(df2.shape[0]):
                filename = path+df1[j,0]+"_"+df1[j,2]+"_"+df2[k,0]+".parquet"
                if os.path.exists(filename) ==1 :
                    df=pq.read_table(filename).to_pandas()
                    if(df.shape[0] == 18000):
                        df["uu"] = df.index
                        dg = df.values[8999,5]

                        df = df.reset_index()
                        df = df["f"]
                        st.append(df2[k,0])
                        break
            print(k)

            for i in range(k+1,df2.shape[0]):
                filename = path+df1[j,0]+"_"+df1[j,2]+"_"+df2[i,0]+".parquet"
                if os.path.exists(filename) ==1 :

                    dt=pq.read_table(filename).to_pandas()  
                    dt = dt.dropna(how = "any",axis = 0)
                    if(dt.shape[0] == 18000):
                        dt = dt.reset_index()
                        dt = dt["f"]
                        df = pd.concat([df,dt],axis = 1)
                        # df['uu'] = df.index
                        st.append(df2[i,0])
                        # print(df.values[-1,5])

      
                        # print(df.shape)
            df.columns = st
                

            print(df.head())
            print(df.shape)
            print(dg)
            savename="y="+str(dg.year)+"_m="+str(dg.month)+"_d="+str(dg.day)+"_h="+str(dg.hour)+"_min="+str(dg.minute)
            event_type = df1[j,2]
            path3 = '../ex_data/freq/'

            if(df.shape[0]!=0):
                df.to_parquet(path3+event_type+"_"+savename+".parquet") 
                print(savename + " save done")
        except:
            print("error")
        # print(dg.year)
        # print(dg.month)
        # print(dg.day)

        # print(dg.hour)
        # print(dg.minute)
        # print(dg.second)


def tt(mm):
    path = "y2017_m"+str(mm)+"/"
    fname = "stat-y2017_m"+str(mm)+".csv"
    df1 = pd.read_csv(fname).values
    df2 = pd.read_csv("cc.csv").values

    for j in range(1):
        print(df1[j,0])
        v = 0
        v2 = 0        
        st = []

        for k in range(5):
            filename = path+df1[j,0]+"_"+df1[j,2]+"_"+df2[k,0]+".parquet"
            if os.path.exists(filename) ==1 :
                df=pq.read_table(filename).to_pandas()
                if(df.shape[0] == 18000):
                    # df["uu"] = df.index
                
   
                    print(df.head())
def save_v(mm):
    path = "y2016_m"+str(mm)+"/"
    fname = "stat-y2016_m"+str(mm)+".csv"
    df1 = pd.read_csv(fname).values
    df2 = pd.read_csv("cc.csv").values

    for j in range(df1.shape[0]):
        print(df1[j,0])
        v = 0
        v2 = 0        
        st = []
        try:
            for k in range(df2.shape[0]):
                filename = path+df1[j,0]+"_"+df1[j,2]+"_"+df2[k,0]+".parquet"
                if os.path.exists(filename) ==1 :
                    df=pq.read_table(filename).to_pandas()
                    if(df.shape[0] == 18000):
                        df["uu"] = df.index
                        dg = df.values[8999,5]

                        df = df.reset_index()
                        df = df["vp_m"]
                        st.append(df2[k,0])
                        break
            print(k)

            for i in range(k+1,df2.shape[0]):
                filename = path+df1[j,0]+"_"+df1[j,2]+"_"+df2[i,0]+".parquet"
                if os.path.exists(filename) ==1 :

                    dt=pq.read_table(filename).to_pandas()  
                    dt = dt.dropna(how = "any",axis = 0)
                    if(dt.shape[0] == 18000):
                        dt = dt.reset_index()
                        dt = dt["vp_m"]
                        df = pd.concat([df,dt],axis = 1)
                        # df['uu'] = df.index
                        st.append(df2[i,0])
                        # print(df.values[-1,5])

      
                        # print(df.shape)
            df.columns = st
                

            # print(df.head())
            print(df.shape)
            print(dg)
            savename="y="+str(dg.year)+"_m="+str(dg.month)+"_d="+str(dg.day)+"_h="+str(dg.hour)+"_min="+str(dg.minute)
            event_type = df1[j,2]
            path3 = '../ex_data/vpm/'

            if(df.shape[0]!=0):
                df.to_parquet(path3+event_type+"_"+savename+".parquet") 
                print(savename + " save done")
        except:
            print("error")
        # print(dg.year)
        # print(dg.month)
        # print(dg.day)

        # print(dg.hour)
        # print(dg.minute)
        # print(dg.second)

def save_i(mm):
    path = "y2016_m"+str(mm)+"/"
    fname = "stat-y2016_m"+str(mm)+".csv"
    df1 = pd.read_csv(fname).values
    df2 = pd.read_csv("cc.csv").values

    for j in range(df1.shape[0]):
        print(df1[j,0])
        v = 0
        v2 = 0        
        st = []
        try:
            for k in range(df2.shape[0]):
                filename = path+df1[j,0]+"_"+df1[j,2]+"_"+df2[k,0]+".parquet"
                if os.path.exists(filename) ==1 :
                    df=pq.read_table(filename).to_pandas()
                    if(df.shape[0] == 18000):
                        df["uu"] = df.index
                        dg = df.values[8999,5]

                        df = df.reset_index()
                        df = df["ip_m"]
                        st.append(df2[k,0])
                        break
            print(k)

            for i in range(k+1,df2.shape[0]):
                filename = path+df1[j,0]+"_"+df1[j,2]+"_"+df2[i,0]+".parquet"
                if os.path.exists(filename) ==1 :

                    dt=pq.read_table(filename).to_pandas()  
                    dt = dt.dropna(how = "any",axis = 0)
                    if(dt.shape[0] == 18000):
                        dt = dt.reset_index()
                        dt = dt["ip_m"]
                        df = pd.concat([df,dt],axis = 1)
                        # df['uu'] = df.index
                        st.append(df2[i,0])
                        # print(df.values[-1,5])

      
                        # print(df.shape)
            df.columns = st
                

            # print(df.head())
            print(df.shape)
            print(dg)
            savename="y="+str(dg.year)+"_m="+str(dg.month)+"_d="+str(dg.day)+"_h="+str(dg.hour)+"_min="+str(dg.minute)
            event_type = df1[j,2]
            path3 = '../ex_data/ipm/'

            if(df.shape[0]!=0):
                df.to_parquet(path3+event_type+"_"+savename+".parquet") 
                print(savename + " save done")
        except:
            print("error")

def save_ia(mm):
    path = "y2016_m"+str(mm)+"/"
    fname = "stat-y2016_m"+str(mm)+".csv"
    df1 = pd.read_csv(fname).values
    df2 = pd.read_csv("cc.csv").values

    for j in range(df1.shape[0]):
        print(df1[j,0])
        v = 0
        v2 = 0        
        st = []
        try:
            for k in range(df2.shape[0]):
                filename = path+df1[j,0]+"_"+df1[j,2]+"_"+df2[k,0]+".parquet"
                if os.path.exists(filename) ==1 :
                    df=pq.read_table(filename).to_pandas()
                    if(df.shape[0] == 18000):
                        df["uu"] = df.index
                        dg = df.values[8999,5]

                        df = df.reset_index()
                        df = df["ip_a"]
                        st.append(df2[k,0])
                        break
            print(k)

            for i in range(k+1,df2.shape[0]):
                filename = path+df1[j,0]+"_"+df1[j,2]+"_"+df2[i,0]+".parquet"
                if os.path.exists(filename) ==1 :

                    dt=pq.read_table(filename).to_pandas()  
                    dt = dt.dropna(how = "any",axis = 0)
                    if(dt.shape[0] == 18000):
                        dt = dt.reset_index()
                        dt = dt["ip_a"]
                        df = pd.concat([df,dt],axis = 1)
                        # df['uu'] = df.index
                        st.append(df2[i,0])
                        # print(df.values[-1,5])

      
                        # print(df.shape)
            df.columns = st
                

            # print(df.head())
            print(df.shape)
            print(dg)
            savename="y="+str(dg.year)+"_m="+str(dg.month)+"_d="+str(dg.day)+"_h="+str(dg.hour)+"_min="+str(dg.minute)
            event_type = df1[j,2]
            path3 = '../ex_data/ipa/'

            if(df.shape[0]!=0):
                df.to_parquet(path3+event_type+"_"+savename+".parquet") 
                print(savename + " save done")
        except:
            print("error")

def save_va(mm):
    path = "y2016_m"+str(mm)+"/"
    fname = "stat-y2016_m"+str(mm)+".csv"
    df1 = pd.read_csv(fname).values
    df2 = pd.read_csv("cc.csv").values

    for j in range(df1.shape[0]):
        print(df1[j,0])
        v = 0
        v2 = 0        
        st = []
        try:
            for k in range(df2.shape[0]):
                filename = path+df1[j,0]+"_"+df1[j,2]+"_"+df2[k,0]+".parquet"
                if os.path.exists(filename) ==1 :
                    df=pq.read_table(filename).to_pandas()
                    if(df.shape[0] == 18000):
                        df["uu"] = df.index
                        dg = df.values[8999,5]

                        df = df.reset_index()
                        df = df["vp_a"]
                        st.append(df2[k,0])
                        break
            print(k)

            for i in range(k+1,df2.shape[0]):
                filename = path+df1[j,0]+"_"+df1[j,2]+"_"+df2[i,0]+".parquet"
                if os.path.exists(filename) ==1 :

                    dt=pq.read_table(filename).to_pandas()  
                    dt = dt.dropna(how = "any",axis = 0)
                    if(dt.shape[0] == 18000):
                        dt = dt.reset_index()
                        dt = dt["vp_a"]
                        df = pd.concat([df,dt],axis = 1)
                        # df['uu'] = df.index
                        st.append(df2[i,0])
                        # print(df.values[-1,5])

      
                        # print(df.shape)
            df.columns = st
                

            # print(df.head())
            print(df.shape)
            print(dg)
            savename="y="+str(dg.year)+"_m="+str(dg.month)+"_d="+str(dg.day)+"_h="+str(dg.hour)+"_min="+str(dg.minute)
            event_type = df1[j,2]
            path3 = '../ex_data/vpa/'

            if(df.shape[0]!=0):
                df.to_parquet(path3+event_type+"_"+savename+".parquet") 
                print(savename + " save done")
        except:
            print("error")


def main():
    # y2016_m5
    # list3 = ["y2016_m5","y2016_m8","y2016_m9","y2016_m10","y2016_m11","y2016_m12"]
    # list4 = ["y2017_m3","y2017_m4","y2017_m5","y2017_m6","y2017_m7","y2017_m8","y2017_m9"]
    # read_all()
    # tt(12)
    for i in range(1,13):
        save_va(i)
        save_ia(i)
        # save_v(i)
        # save_i(i)
        # save_f(i)
    # find_all_pmu()

if __name__ == '__main__':  

    main()
