

# import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import math
import pandas as pd
import time  
from sklearn import metrics  
import pickle as pickle  
import timeit
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import os
import sys
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


   
def classit2(data,t1,t3,t5,t7,t8,data2,t9):

    # list5 =[0,2,4,6,7]
    
    temp = data[:,0]
    for i in range(temp.shape[0]):
        if(temp[i]>t1):
            # vpm
            return 1

    temp = data[:,2]
    for i in range(temp.shape[0]):
        if(temp[i]>t3):
            return 1
          
    temp = data[:,4]
    for i in range(temp.shape[0]):
        if(temp[i]>t5):
            return 1
            
    temp = data[:,6]
    for i in range(temp.shape[0]):
        if(temp[i]>t7):
            # vpm
            return 1               
    # print("temp.shape",temp.shape)
    temp = data[:,7]
    for i in range(temp.shape[0]):
        if(temp[i]>t8):
            # vpm
            return 1   
            
    for j in range(23):
        temp = data2[j,:]
        for i in range(temp.shape[0]):
            if(np.abs(temp[i])>t9):
                return 1                          
    return 0




def deal(ii):
    # X_S1.pickle
    path1 = "2016/"
    p1 = open(path1 +'X_S'+str(ii)+'.pickle',"rb")
    pk1 = pickle.load(p1)    
    print("pk1.shape",pk1.shape)
    
   
    # ii =1
    p1 = open(path1 +'X_S'+str(ii)+'_rocof_6.pickle',"rb")
    pk2 = pickle.load(p1)  
    print("pk2.shape",pk1.shape)
    st=[]
    ll =[0.8939229464778622,
        0.03429627874409605, 
        1.344868701091236, 
        0.19396349528008003, 
        0.036158937893934935]
        
    t1 = ll[0]
    t3 = ll[1]
    t5 = ll[2]
    t7 = ll[3]
    t8 = ll[4]
    
    for i in range(pk1.shape[0]):
        temp = pk1[i]
        temp2 = pk2[i,:,:,0]
        res = classit2(temp,t1,t3,t5,t7,t8,temp2,0.15)
        st.append(res)
    df = pd.DataFrame(st)
    dt = pd.read_csv("2016/y_S"+str(ii)+"_vp_m_6.csv")
    df["word"] =dt["0"]
    df.to_csv(str(2016)+"-"+str(ii)+".csv",index = None)
    # print(st)
    
def deal2(ii):
    # X_S1.pickle
    path1 = "2017/"
    p1 = open(path1 +'X_S'+str(ii)+'.pickle',"rb")
    pk1 = pickle.load(p1)    
    print("pk1.shape",pk1.shape)
    
   
    # ii =1
    p1 = open(path1 +'X_S'+str(ii)+'_rocof_6.pickle',"rb")
    pk2 = pickle.load(p1)  
    print("pk2.shape",pk1.shape)
    st=[]
    ll =[0.8939229464778622,
        0.03429627874409605, 
        1.344868701091236, 
        0.19396349528008003, 
        0.036158937893934935]
        
    t1 = ll[0]
    t3 = ll[1]
    t5 = ll[2]
    t7 = ll[3]
    t8 = ll[4]
    
    for i in range(pk1.shape[0]):
        temp = pk1[i]
        temp2 = pk2[i,:,:,0]
        res = classit2(temp,t1,t3,t5,t7,t8,temp2,0.15)
        st.append(res)
    df = pd.DataFrame(st)
    dt = pd.read_csv("2017/y_S"+str(ii)+"_vp_m_6.csv")
    df["word"] =dt["0"]
    df.to_csv(str(2017)+"-"+str(ii)+".csv",index = None)
    # print(st)
    

    
mm = ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
def change_file():
    df = pd.read_csv("2016_2.csv")
    # df.shape[0]
    st = []
    for i in range(df.shape[0]):
        temp = df.loc[i,"Start"]
        x1 = temp.split(" ")[0]
        x2 = x1.split("/")
        st.append(x2[2]+"_"+mm[int(x2[0])]+"_"+x2[1])
        # print(x2)
        # print(type(int(x2[0])))
        # print(int(x2[0]))
    st = np.array(st)
    df["nn"] = st
    df["id"] = df.groupby(["nn"]).cumcount()
    st =[]
    for i in range(df.shape[0]):
        temp1 = df.loc[i,"nn"]
        temp2 = df.loc[i,"id"]
        st.append(temp1+"_"+str(temp2))
    st = np.array(st)
    
    df["new"] = st
    df.to_csv("2016new.csv",index = None)
    print(df.head())
    
    
def change_file2():
    df = pd.read_csv("2017_2.csv")
    # df.shape[0]
    st = []
    for i in range(df.shape[0]):
        temp = df.loc[i,"Start"]
        x1 = temp.split(" ")[0]
        x2 = x1.split("/")
        st.append(x2[2]+"_"+mm[int(x2[0])]+"_"+x2[1])
        # print(x2)
        # print(type(int(x2[0])))
        # print(int(x2[0]))
    st = np.array(st)
    df["nn"] = st
    df["id"] = df.groupby(["nn"]).cumcount()
    st =[]
    for i in range(df.shape[0]):
        temp1 = df.loc[i,"nn"]
        temp2 = df.loc[i,"id"]
        st.append(temp1+"_"+str(temp2))
    st = np.array(st)
    
    df["new"] = st
    df.to_csv("2017new.csv",index = None)
    print(df.head())

def rd(ii):
    print(ii)
    df = pd.read_csv("res/"+str(2016)+"-"+str(ii)+".csv")
    df.columns = ["label","new"]
    print(df.shape)
    dt = pd.read_csv("2016new.csv")
    
    df2 = pd.merge(df,dt, on=["new"])
    print(df2.shape)
    print(df2.head())
    df2.pop("Event")
    df2.pop("nn")
    df2.pop("id")
    df2.to_csv("res/2016--"+str(ii)+".csv",index = 0)
    
    
    
def rd2(ii):
    print(ii)
    df = pd.read_csv("res/"+str(2017)+"-"+str(ii)+".csv")
    df.columns = ["label","new"]
    # print(df.shape)
    dt = pd.read_csv("2017new.csv")
    
    df2 = pd.merge(df,dt, on=["new"])
    # print(df2.shape)
    # print(df2.head())
    df2.pop("Event")
    df2.pop("nn")
    df2.pop("id")
    df2.to_csv("res/2017--"+str(ii)+".csv",index = 0)
    
def gen_all():
    df1 = pd.read_csv("res/"+str(2016)+"--"+str(1)+".csv")
    for ii in range(2,14):
        if(os.path.exists("res/"+str(2016)+"--"+str(ii)+".csv")):
            df = pd.read_csv("res/"+str(2016)+"--"+str(ii)+".csv")
            df1 = pd.concat([df1,df])
            
    for ii in range(1,14):
        if(os.path.exists("res/"+str(2017)+"--"+str(ii)+".csv")):
            df = pd.read_csv("res/"+str(2017)+"--"+str(ii)+".csv")
            df1 = pd.concat([df1,df]) 

    df1.to_csv("all_eve.csv",index= 0)
    df = df1[df1["label"] == 0]
    df.to_csv("not_detected.csv",index= 0)
    df = df1[df1["label"] == 1]
    df.to_csv("all_detected.csv",index= 0)    
    
def deal3(ii):
    # X_S1.pickle
    path1 = "2017/"

    p1 = open(path1 +'X_S'+str(ii)+'_vp_m_6.pickle',"rb")
    pk2 = pickle.load(p1)    
    print(pk2.shape)
    df = pd.read_csv("2017-"+str(ii)+".csv")
    df.columns = ["label","new"]
    print(df.shape)
    
    res = np.zeros(pk2.shape[0])
    for j in range(pk2.shape[0]):
        for i in range(pk2.shape[1]):
            temp2 = pk2[j,i,:,0]/np.mean(pk2[j,i,:,0])
            for k in range(5400-360):
                temp3 = temp2[k:k+360]
                if(np.abs(np.max(temp3)-np.min(temp3))>0.004):
                    res[j] =1

    dt = df["label"].values
    for j in range(pk2.shape[0]):
        if(res[j]==1):
            dt[j] =1
    df["label"] =dt
    
    df.to_csv("res/"+str(2017)+"-"+str(ii)+".csv",index = None)
        
    
    
def main():
    s1 = timeit.default_timer()  
    # change_file2()
    # for ii in range(1,8):
        # rd(ii)  
    
    # for ii in range(8,13):
        # rd(ii)
    
    # change_file()
    # rd(1)
    gen_all()
    # deal3(12)
    
    
    s2 = timeit.default_timer() 
    print ('Runing time is Hour:',round((s2 -s1)/3600,2))
        

if __name__ == '__main__':  



    
    # get_index()
    main()
    # get_nor()
