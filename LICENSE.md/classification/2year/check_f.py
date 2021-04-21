# This code is to find out the mutilabel events and it statics

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

import pickle
import os
import sys

from datetime import datetime

import timeit



def fun2(ii):

    path1 = 'extracted/'
    list = []

    for line in open(path1 + "a_"+str(ii)+".txt"): 
        list.append(line.replace("\n", ""))
 
    return( len(list))

def cal():
    aa = [283, 336, 410, 442, 475, 403, 263, 283, 355, 439, 325, 526, 314]

    num = []

    for i in range(len(aa)+1):
        sum = 0
        for j in range(i):
            sum+=aa[j]
        num.append(sum)
    print(len(num))
    print(num)
    

def check_muti():
    df = pd.read_csv("muti.csv")
    print(df.shape)
    tt = df['new'].unique()
    print(tt.shape)
    
nums = [0, 283, 619, 1029, 1471, 1946, 2349, 2612, 2895, 3250, 3689, 4014, 4540, 4854]
def fun(ii):
    print("data"+str(ii))
    path1 = 'extracted/'
    list = []


    for line in open(path1 + "a_"+str(ii)+".txt"): 
        list.append(line.replace("\n", ""))
 
    long = len(list)

    print(len(list))
    
    df = pd.read_csv("B_Training1.csv")
    df = df.iloc[(nums[ii-1]):(nums[ii])].values
    
    list2= []
    for i in range(df.shape[0]):
        list2.append(str(df[i][4])+"_"+str(df[i][5])+"_"+sadd(str(df[i][6]))+"_"+str(df[i][7])+"_"+str(df[i][8]))
    print(len(list2))
    
    
    
    # list.sort()
    # list2.sort() # 改变list本身
    
    list3 =[]
    for c in list:
        if(c not in list2):
            list3.append(c)
    print(len(list3))
    print(list3)
    
def fun2():

    df = pd.read_csv("B_Training1.csv")
    df = df.values
    
    list2= []
    for i in range(df.shape[0]):
        list2.append(str(df[i][4])+"_"+str(df[i][5])+"_"+sadd(str(df[i][6]))+"_"+str(df[i][7])+"_"+str(df[i][8].split(" ")[0]))
    print(len(list2))
    
    df2 = pd.read_csv("cc.csv").values
    list3 =[]
    for i in range(df2.shape[0]):
        list3.append(str(df2[i][2]))
    print(len(list3))    
    print(list2[0])
    print(list3[0])
    
    list =[]
    for c in list3:
        if(c not in list2):
            list.append(c)
    print(len(list))
    # print(list)
        
    
def sadd(str):
    # str = "2"
    if(len(str)) == 1:
        return "0"+str
    else:
        return str
            
def check_cc():
    df = pd.read_csv("cc.csv")
    df = df.values
    list = []
    for i in range(0,df.shape[0]):
        if(df[i][0] == 0 and "Line" not in df[i][2]):
            list.append(df[i][2])
        elif(df[i][0] == 1 and "XFMR" not in df[i][2]):
            list.append(df[i][2])
        elif(df[i][0] == 2 and "Freq" not in df[i][2]):
            list.append(df[i][2])
        elif(df[i][0] == 3 and "Osc" not in df[i][2]):
            list.append(df[i][2])
        elif(df[i][0] == 4 and "Line" not in df[i][2]):
            list.append(df[i][2])        
        elif(df[i][0] == 5 and "XFMR" not in df[i][2]):
            list.append(df[i][2])  
    print(len(list))

def remove_z(str):
    if str[0] == '0':
        return str[1]
    else:
        return str
            
def fun3():

    X_new=[]
    y_new=[]
    word =[]
    df = pd.read_csv("muti.csv")
    ff = df.values
    df = df["new"].values.tolist()

    cnt1= 0
    cnt2= 0
    for i in range(len(ff)):
        if(ff[i][5] == 5 and ff[i][1]=="Frequency"):
            cnt1+=1
            
        elif(ff[i][5] == 7 and ff[i][1]=="Frequency"):
            cnt2+=1        

    y = pd.read_csv("cc.csv").values
    s3 =0
    for i in range(len(y)):
        #print(i)
        temp = y[i,2].split("_")
        ww = temp[0]+"_"+temp[1]+"_"+remove_z(temp[2])+"_"+temp[3]
        # print(np.isin(ww,df))

        if(np.isin(ww,df)==False):
        
            if y[i,0]==0:
                y_new.append(0)
                word.append(y[i,2])
                
                
            elif y[i,0]==1:
                y_new.append(1)
                word.append(y[i,2])
        
                
            elif y[i,0]==2:
                y_new.append(0)
                word.append(y[i,2])
            
            elif y[i,0]==3:
                y_new.append(0)
                word.append(y[i,2])
            elif y[i,0]==4:
                y_new.append(0)
                word.append(y[i,2])
            elif y[i,0]==5:
                y_new.append(1)
                word.append(y[i,2])
        else:
            s3+=1
    print(len(y_new))
    print(len(word))
    s1= 0
    s2 =0
    

    
    for i in range(len(y_new)):
        if(y_new[i]== 1):
            s2+=1
        else:
            s1+=1
    print("freq num:",s2)
    print("Non freq num:",s1)
    print("s3:",s3)    

    print("freq of Line + Freq:",cnt1)
    print("freq of Line + Freq + Trans:",cnt2)

#statics based on event logs
def fun4():

    X_new=[]
    y_new=[]
    word =[]
    df = pd.read_csv("muti.csv")
    ff = df.values
    df = df["new"].values.tolist()

    cnt1= 0
    cnt2= 0
    cnt3 = 0
    for i in range(len(ff)):
        # if(ff[i][5] == 7 and ff[i][1]=="Transformer"):
            # cnt1+=1
            
        if(ff[i][5] == 5 and ff[i][1]=="Line"):
            cnt2+=1        

        elif(ff[i][5] == 5 and ff[i][1]=="Frequency"):
            cnt3+=1        

    print("Line of Line + Freq:",cnt2)
    print("Frequency of Line + Freq:",cnt3)    
    # print("Frequency of Line + Freq + Trans:",cnt3)  
def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)


#statics based on event Time
def fun5():

    X_new=[]
    y_new=[]
    word =[]
    df = pd.read_csv("muti.csv")
    list2 = []
    dt = df['Start'].unique()
    df = df[df["v"]==7]
    cnt1= 0
    cnt2= 0
    cnt3 = 0
    # dt.shape[0]
    for i in range(dt.shape[0]):
        temp = df[df["Start"] == dt[i]]
        if(temp.shape[0]>1):
            # temp = temp["Category"].values.tolist()
            # print(type(temp))
            cnt2 +=1
    print(cnt2)
    print(dt.shape)
    # list2 = list(flat(list2))
    # print(list2)
    
    
    # cnt1 = list2.count('Line')
    # cnt2 = list2.count('Transformer')
    # print("Line:"+str(cnt1))
    
    # print("Transformer:"+str(cnt2))
    
def remove_z(str):
    if str[0] == '0':
        return str[1]
    else:
        return str
        
#statics for single label
def fun6():


    X_new=[]
    y_new=[]
    word =[]
    df = pd.read_csv("muti.csv")
    df = df["new"].values.tolist()
    
    y = pd.read_csv("B_Training1.csv").values
    for i in range(len(y)):
        #print(i)
        temp = y[i]
        ww = str(temp[4])+"_"+str(temp[5])+"_"+str(temp[6])+"_"+str(temp[7])
        # print(np.isin(ww,df))

        if(np.isin(ww,df)==False):
        
            if y[i,1]=="Line":

                y_new.append(0)
                
                
            elif y[i,1]=="Transformer":

                y_new.append(1)
        
                
            elif y[i,1]=="Frequency":
                y_new.append(2)

            
            elif y[i,1]=="Oscillation":
                y_new.append(3)
    cnt1 = y_new.count(0)
    cnt2 = y_new.count(1)
    cnt3 = y_new.count(2)
    cnt4 = y_new.count(3)    
    print(len(y_new))
    print("Line:"+str(cnt1))
    print("Transformer:"+str(cnt2))
    print("Frequency:"+str(cnt3))    
    print("Oscillation:"+str(cnt4))


def main():


    fun6()

    # check_cc()
    # fun2()

    # check_muti()
    # list = []
    # for i in range(1,14):
        # fun(i)
    # print(list)
    # cal()
if __name__ == '__main__':  

    main()
