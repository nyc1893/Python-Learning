import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import timeit

path = "time/"
def add_zero(a):
    if(len(a)<2):
        return "0"+a
    else:
        return a
def cc():
    df= pd.read_csv(path + "2016_test.csv")
    dt = df.values
    st = []
    for i in range(df.shape[0]):    
        temp= dt[i,0]
        x= temp.split("_")
        x1 = x[0].split("=")
        x2 = x[1].split("=")
        x3 = x[2].split("=")
        x4 = x[3].split("=")
        x5 = x4[1].split(".")
        
        # print(x1[1],x2[1],x3[1],x5[0])
        st.append(x1[1]+"/"+x2[1]+"/2016 "+ x3[1]+":"+add_zero(x5[0]))
        

    df["time"] = st
    print(df.head())
    dt = df.values
    # df.shape[0]
    st1 = []
    st2 = []
    for i in range(df.shape[0]):
        temp1= dt[i,1]
        temp2= dt[i,2]
        temp1 = -(3600 - temp1 )/60
        temp2 = -(3600 - temp2 )/60
        st1.append(temp1)
        st2.append(temp2)
        # print(temp1)
        # print(temp2)
    df["line_index"] = st1
    df["freq_index"] = st2
    
    df = df.dropna(axis =1,how = "any")
    df.pop("f_name")
    print(df.head())
    
    df.to_csv("2016_2.csv",index = None)

def change_label():
    path = "orignal/"
    df= pd.read_csv(path+"2016B_test.csv")
    
    dt1 = df[df["Event"] =="Freq"]
    dt2 = df[df["Event"] =="Osc"]
    
    print(dt1.shape)
    print(dt2.shape)
    
    dt1 = dt1["Start"].values.tolist()
    dt2 = dt2["Start"].values.tolist()
    
    df= pd.read_csv("2016_2.csv")
    ind1 = df.loc[df.time.isin(dt1)].index
    ind2 = df.loc[df.time.isin(dt2)].index
    
    print(len(ind1), len(ind2))
    df.RF[ind1] = 2
    df.RF[ind2] = 3
    df.to_csv("2016_3.csv",index = None)
    
def deal_time():
    res =[]
    df= pd.read_csv("2016_3.csv")
    df['time']=pd.to_datetime(df['time'], format='%m/%d/%Y %H:%M')
    st =[]
    for i in range(df.shape[0]):
        if(df.loc[i,"RF"] == 2):
            st.append(df.loc[i,"freq_index"])
        else:
            st.append(df.loc[i,"line_index"])

    df["ind"] = st
    
    for i in range(df.shape[0]):
        temp = df.loc[i,"time"] +timedelta(seconds = df.loc[i,"ind"])
        res.append(temp)
        
    df["c_time"] = res
    df['c_time'] = df['c_time'].dt.strftime('%Y/%m/%d %H:%M:%S.%f')
    print(df.head())
    
    st2=[]
    print(df["RF"].dtypes)
    for i in range(df.shape[0]):
        temp = df.loc[i,"RF"] 
        if(temp == 0):
            st2.append("Line")
        elif(temp == 1):
            st2.append("Transformer")
        elif(temp == 2):
            st2.append("Frequency ") 
        elif(temp == 3):
            st2.append("Oscillation")   
    df["event"] = st2
    df.to_csv("2016_4.csv",index = None)
def main():
    deal_time()
    # change_label()



if __name__ == "__main__":
    main()


