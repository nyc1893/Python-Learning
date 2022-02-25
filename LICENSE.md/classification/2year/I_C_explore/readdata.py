# This code is to find out the mutilabel events and it statics

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

import pickle
import os
import sys

from datetime import datetime

import timeit



def read():

    name2 = 'y2016_m4'
    path = "./"+name2+"/"
    path2 = "./" + name2
    rootpath =os.listdir(path2)
    
    rootpath.sort(key= lambda x:str(x))
    nn = len(rootpath)
    sum = 0
    list2 = []
    df1= pd.read_csv("all3.csv")
    df1 = df1["eve_name"].values.tolist()    
    
    df2 = pd.read_csv("cc.csv")
    df2= df2["30fps"].values.tolist()   
    # PMU_list_1 = PMU_list['30fps']
    st =[]
    features=[]
    labels=[]
    flag = 1
    j =0
    label_word=[]
    num_time_sample = 5400
    min = num_time_sample
    
    # nn = 190
    for i in range(0,nn):
        filename = path + rootpath[i]
        x = rootpath[i].split("_")
        ww = x[0]+"_"+x[1]+"_"+x[2]+"_"+x[3]
        x2 = str(x[5]).split(".")
        # print(x)
        pmu = x2[0]
        if os.path.exists(filename) ==1 and np.isin(ww,df1)==True and np.isin(pmu,df2)==True:
            # print(ww)
            df=pq.read_table(filename).to_pandas()
            df = df.dropna(axis=0,how='all')  
            if (df['f'][4*1800:7*1800].isna().sum()!=0):
            # or df['vp_m'].isna().sum()!=0 or df['vp_m'].isna().sum()!=0):
                # print(df['f'].isna().sum())
                # df.to_csv("test.csv",index =None)
                print("flag1")
                
            if df['f'][4*1800:7*1800].shape[0] == num_time_sample:
                st.append(df['f'][4*1800:7*1800]) 
                
            if(min>df['f'][4*1800:7*1800].shape[0]):
                min = df['f'][4*1800:7*1800].shape[0]
                # print("flag2")
                print(rootpath[i])
                
        if(len(st)!=0 and len(st)%148 ==0):

            j=j+1
            print("j =",j)
            st = np.array(st)
            if (min==num_time_sample and flag ==1): 
                if(st.shape[0]==148):
                    features.append(st)
                
                    event_name = x[4]
                    label_word.append(ww)
                    if 'Line' in event_name:
                        labels.append(0)
                            
                    elif 'Generator' in event_name:
                        labels.append(1)
                              
                    elif 'Transformer' in event_name: 
                        labels.append(2)
                    elif 'Oscillation' in event_name: 
                        labels.append(3) 
            st=[]       
            min=num_time_sample
            flag = 1                        
    features=np.array(features) 
    print(features.shape)
    
    

def read2():
    res =[]
    st =[]
    for mm in range(1,13):
        name2 = 'y2017_m'+str(mm)
        path = "./"+name2+"/"
        path2 = "./" + name2
        rootpath =os.listdir(path2)
        
        rootpath.sort(key= lambda x:str(x))
        nn = len(rootpath)
        sum = 0
        list2 = []
        df1= pd.read_csv("all3.csv")
        df1 = df1["eve_name"].values.tolist()    
        
        df2 = pd.read_csv("cc.csv")
        df2= df2["30fps"].values.tolist()   
        # PMU_list_1 = PMU_list['30fps']
        
        num_time_sample = 5400
        min = num_time_sample



        
        for i in range(0,nn):
            try:
                filename = path + rootpath[i]
                x = rootpath[i].split("_")
                ww = x[0]+"_"+x[1]+"_"+x[2]+"_"+x[3]
                x2 = str(x[5]).split(".")
    # /scratch/share/ic_c/y2016_m1/2016_Jan_2_0_Line_C529.parquet
                if os.path.exists(filename) ==1 and ".parquet" in filename:
                    df=pq.read_table(filename).to_pandas()
                    df["ind"] = df.index
                    res.append(df.iloc[9000-1].values[5])
                    st.append(filename)
                    print(filename)
            except:
                print("error!")
    res = pd.DataFrame(res)
    res["fname"] = st
    res.to_csv("2017-check.csv",index= None)



def fuu():
    num_samples,h,w=features.shape
    features=features.reshape(num_samples, h, w,1)

    
    
    name = list[6]
    kk = 1
    path3 = 'pickleset1/'

    pickle_out = open(path3 + "X_"+name2+"_"+str(name)+"_3.pickle","wb")
    pickle.dump(features, pickle_out, protocol=2)
    pickle_out.close() 
    
    b = np.array(ename)
    #save labels for all the events
    labels=np.array(labels)

    labels = labels.reshape(num_samples,-1)
    # label_word = label_word.reshape(num_samples,-1)
    # b = b.reshape(num_samples,-1)
    # print(labels.shape)
    # print(b.shape)
    
    labels=np.concatenate((labels, label_word), axis=1)
    # labels=np.concatenate((labels, b), axis=1)
    
    yy = pd.DataFrame(labels)
    yy.to_csv(path3 +"y_"+name2+"_"+str(name)+"_3.csv",index =None)
    print(yy.head())    
    
    
def main():
    # y2016_m5
    # list3 = ["y2016_m5","y2016_m8","y2016_m9","y2016_m10","y2016_m11","y2016_m12"]
    # list4 = ["y2017_m3","y2017_m4","y2017_m5","y2017_m6","y2017_m7","y2017_m8","y2017_m9"]

    read2()
    # ss = "y2016_m"
    # for i in range(1,12+1):
        # read2(ss+str(i))    

    # ss = "y2017_m"
    # for i in range(2,11+1):
        # read2(ss+str(i))   
        
    # ss = "y2017_m8"
    # get_188(ss)
if __name__ == '__main__':  

    main()
