
# This is for Feq CWT^2 use
import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import math
import pandas as pd

import pywt
import pickle
import timeit
import datetime
import heapq

from scipy.stats import pearsonr,spearmanr,kendalltau
from scipy import signal 
import os
import sys
from statistics import variance 


# import datapick


start = timeit.default_timer()





def Modified_Z(data):
    c = 1.4826
    median = np.median(data)
    # print(median)
    # print("median.shape",median.shape)
    dev_med = np.array(data) -median
    # print("dev_med.shape",dev_med.shape)
    mad = np.median(np.abs(dev_med))
    if mad!=0:
        
        z_score = dev_med/(c*mad)
    else : 
        df = pd.DataFrame(data)
        meanAD = df.mad().values
        z_score =  dev_med/(1.253314*meanAD)
        
    return z_score
    

def remove_z(str):
    if str[0] == '0':
        return str[1]
    else:
        return str
        
def rm3(X,y):
    """
    THIS FUNCTION REMOVES THE PLANNED EVENTS FROM THE EVENT DATASET
    """
    
    X_new=[]
    y_new=[]
    word =[]
    df = pd.read_csv("../../../muti.csv")
    df = df["new"].values.tolist()
    # print(len(df))
    # print(df[:5])
    for i in range(len(y)):
        #print(i)
        temp = y[i,2].split("_")
        ww = temp[0]+"_"+temp[1]+"_"+remove_z(temp[2])+"_"+temp[3]
        # print(np.isin(ww,df))

        if(np.isin(ww,df)==False):
        
            if y[i,0]==0:
                y_new.append(0)
                X_new.append(X[i,:,:,:])
                word.append(y[i,0])
                
                
            elif y[i,0]==1:
                y_new.append(1)
                X_new.append(X[i,:,:,:])
                word.append(y[i,0])
        
                
            elif y[i,0]==2:
                y_new.append(2)
                X_new.append(X[i,:,:,:])
                word.append(y[i,0])
            
            elif y[i,0]==3:
                y_new.append(3)
                X_new.append(X[i,:,:,:])
                word.append(y[i,0])
            elif y[i,0]==4:
                y_new.append(0)
                X_new.append(X[i,:,:,:])
                word.append(y[i,0])
            elif y[i,0]==5:
                y_new.append(1)
                X_new.append(X[i,:,:,:])
                word.append(y[i,0])
            
        
        
    return  np.array(X_new), np.array(y_new)

def rd_rof(ii):
    path1 = '../pickleset1/'
    list = ['rocof','vp_m','ip_m','f']
    if ii!=12:
        p1 = open(path1 +'X_S'+str(ii)+'_'+str(list[0])+'_6.pickle',"rb")
        pk3  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[0])+'_6.csv')
        pk1 = pickle.load(p1)
    else:
        p1 = open(path1 +'X_S'+str(ii)+'_'+str(list[0])+'_61.pickle',"rb")
        pk3  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[0])+'_61.csv')      
        p2 = open(path1 +'X_S'+str(ii)+'_'+str(list[0])+'_62.pickle',"rb")
        pk32  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[0])+'_62.csv')  
        
        pk1 = pickle.load(p1)
        pk2 = pickle.load(p2)
        # print(pk1.shape)
        # print(pk2.shape)       
        
        pk1 = np.concatenate((pk1, pk2), axis=0)
        pk3 = pd.concat([pk3, pk32])
    

    X_train = pk1
    y_train = pk3.values

    X_train, y_train  = rm3(X_train, y_train)
    print("dataset No. "+str(ii))
    print(X_train.shape)
    print(y_train.shape)
    return X_train, y_train        


    
def choose(X):

    nums = []
    for i in range(0,23):
        temp = X[i,:,0]
        nums.append(np.max(temp)-np.min(temp))
    # nums.sort()
    # print(nums)
    max_num_index_list = map(nums.index, heapq.nlargest(1, nums))

    ll = list(max_num_index_list)
    print("Largest dif ref:",ll[0])
    return np.squeeze(X[ll[0],:,0]),ll[0]
    
def pearsonrSim(x,y):
    '''
    皮尔森相似度
    '''
    return pearsonr(x,y)[0]
 
def cosSim(x,y):
    '''
    余弦相似度计算方法
    '''
    tmp=sum(a*b for a,b in zip(x,y))
    non=np.linalg.norm(x)*np.linalg.norm(y)
    return round(tmp/float(non),3)
     
    
def cal_two(y1,y2):
    zz = Modified_Z(y1)
    z2 = Modified_Z(y2)
    ind = np.argmax(zz)
    ind2 = np.argmax(z2)
    
    # print("ind",ind)
    # print("ind2",ind2)
    if(np.abs(ind - ind2)<100):
        k2 = min(min(ind,ind2),120)
        k3 = min(min(y1[ind:].shape[0],y2[ind2:].shape[0]),120)
        
        n1 = np.append(y1[ind-k2:ind],y1[ind:ind+k3])
        n2 = np.append(y2[ind2-k2:ind2],y2[ind2:ind2+k3])

        k = cal(n1,n2,len(n1))
    else:
        k = cal(y1,y2,min(len(y1),len(y2)))
    # print("simularity: ",k)
    return k    
    
    
def find_err():
    path = "ml/"
    df2 = pd.read_csv(path+"te.csv")
    df3 = pd.read_csv(path+"X_val.csv")
    df3 = df3[["S","No"]]
    df3 = pd.concat([df3,df2],axis =1)
    ind= df3[df3["label"]!=df3["RF"]].index
    df3 = df3.iloc[ind] 
    print(df3.head())
    print(ind)
    df3.to_csv(path+"err.csv",index =None)
    
    
def pack_ploterr():    
    df = pd.read_csv("ml/err.csv")
    df[["S"]] = df[["S"]].astype(np.int32)
    df[["No"]] = df[["No"]].astype(np.int32)

    word = ""
    
    for ii in range(1,14):
        dt = df[df["S"]==ii]
        print(dt.shape[0])
        if(dt.shape[0]>0):
            
            list = dt['No'].tolist()
            print(list)
            plot_err(ii,list)
            
    # for i in range(0,df.shape[0]):
        # word = "Non-Freq"
        # if(df[i,5] == 1):
            # word="Freq"
        # elif(df[i,5] != 1):
            # word="Non-Freq"           
        # plot_err(df[i,0],df[i,1],word)
    

    
def plot_err(ii,kk):
    # ii = 1
    # vp,rof,df1,vp2,rof2,df2 = load_data(ii)
    X_train, y_train  = rd_rof(ii)
    # X_train2, _  = datapick.rd_vpm(ii)
    
    # vp.shape[0]

    list = []
    list2 = []
    list3 = []
    # y = df1["label"].values[j]

    
    for j in kk:
        y = y_train[j]
        
        plt.figure( figsize=(20,5))

        ref,ll = choose(X_train[j,:,:,:])
        list_ind = []    
    
        for i in range(0,23):
            x = np.squeeze(X_train[j,i,:,:])
            # x2= np.squeeze(X_train2[j,i,:,:])
            # print(x.shape)
            zz = Modified_Z(x)
            ind = np.argmin(zz)
            list_ind.append(ind)
            
            # plt.subplot(8,6,2*i+1)
            plt.subplot(4,6,i+1)
            # cc = x[3600-120:3600+180]
            plt.plot(range(len(x)),x,zorder=0)
            plt.scatter(ind,x[ind],marker='p',c='',edgecolors='r',zorder=10)
            
            # plt.plot(range(len(x)),a,zorder=0)
            # plt.plot(range(len(x)),b,zorder=1)
            list.append(round(pearsonrSim(ref,x),3))
            a,b,c = cal_dif(x)

            list2.append(a)
            list3.append(b)
            # plt.xlabel(" sim:"+ str(round(cal_two(ref,x),3))+" dif+:"+str(a)+";"+str(b))
            plt.ylabel("rocof")
            
            
            # plt.subplot(8,6,2*i+2)
            # plt.plot(range(len(x2)),x2,zorder=0)
            
    # confine(list_ind)
        plt.title('real label = '+str(y)  ) 
        # +" pred: "+word
        # plt.subplot(8,6,48)
        plt.subplot(4,6,24)
        plt.xlabel("pearsonrSim avg: "+ str(round(np.mean(list),3)))
        
        temp = np.squeeze(X_train[j,ll,:,:])
        # print("Sim_avg: ",np.mean(list))
        # plt.title('ref:'+str(round(np.max(temp)-np.min(temp),2))+info(list2,list3)) 
        plt.tight_layout()         
        plt.savefig("gen/err_"+str(ii)+"_"+str(j)) 


def info(list2,list3):
    a1 = str(round(np.mean(list2),2))
    a2 = str(round(np.mean(list3),2))
    
    b1 = str(round(np.std(list2),2))
    b2 = str(round(np.std(list3),2))
    return a1,a2,b1,b2
    # return " mean dif+:"+a1+";"+a2+" std:"+b1+";"+b2
    
    
def confine(list):
    duan = np.zeros(100)
    for j in range(0,len(list)):
        for i in range(0,100):
            if list[j]<360*(i+1) and list[j]>360*(i):
                duan[i]+=1
    # print(duan)
        
    list_a = duan.tolist()
    res = list_a.index(max(list_a))
    point = np.median(np.array(list))
    # print(res)
    return res,int(point)    
    

    
def cal_confine(ind,x):
    x = np.squeeze(x)
    # up = []
    # down =[]
    # for i in range(0,30):   
        # temp = x[360*(i):360*(i+1)]
        # up.append(np.max(temp))
        # down.append(np.min(temp))
    # avg_up = np.mean(up)
    temp = x[360*(ind):360*(ind+1)]
    # sum = np.sum(x[360*(ind):360*(ind+1)])
    return np.abs(np.min(temp)/np.max(temp))
    

    
    
    
def save_freq(ii):
    # ii = 1
    # vp,rof,df1,vp2,rof2,df2 = load_data(ii)
    X_train, y_train  = rd_rof(ii)
    # X_train2, _  = datapick.rd_vpm(ii)
    
    st =[]
    for j in range(0,X_train.shape[0]):
        list = []
        list2 = []
        list3 = []
        list_ind = []
        list_mm = []
        list4 = []
        list5 = []
        flag =0
        kk = np.zeros(12)
        # y = df1["label"].values[j]
        y = y_train[j]



        ref,ll = choose(X_train[j,:,:,:])
        # ref = ref[3600-120:3600+180]
        # print("ref.shape",ref.shape)
        for i in range(0,23):
            x = np.squeeze(X_train[j,i,:,:])
            if(x.shape[0]!=36000):
                flag =1
                break
                
            # x2 = np.squeeze(X_train2[j,i,:,:])
            
            list.append(round(pearsonrSim(ref,x),3))
            a,b,mm = cal_dif(x)

            list2.append(a)
            list3.append(b)
            list_mm.append(mm)
            zz= Modified_Z(x)
            # z2= Modified_Z(x2)
            ind = np.argmin(zz)
            # ind2= np.argmin(x2)
            list_ind.append(ind)
            # list_ind2.append(ind2)
        if(flag == 0):
            kk[0] = (np.max(ref)-np.min(ref))    
            kk[1] = round(np.min(list),3)
            kk[2],kk[3],kk[4],kk[5] = info(list2,list3)
            kk[6] = y
            kk[7] = ii
            kk[8] = j
            
            ind,point = confine(list_ind)
            # ind2 =confine(list_ind2)
            tsum= 0
            for i in range(0,23):
                x = np.squeeze(X_train[j,i,:,:])        
                list4.append(cal_confine(ind,x))
                
                if(point>180 and point<36000-180):
                    temp = x[point-180:point+180]
                elif(point>36000-180):
                    temp = x[-360:]
                elif(point<180 ):
                    temp = x[:360]
                tsum = np.sum(temp-list_mm[i])
                list5.append(tsum)
                
            kk[9] = round(np.mean(list4),3)
            kk[10] = round(np.mean(list5),3)
            kk[11] = round(np.max(list5),3)
            # kk[10] = compare_mead(list_ind,list_ind2)
            st.append(kk)
            
    df = pd.DataFrame(st)
    df.columns = ["ref","sim","upBar","downBar","upstd","downstd","label","S","No","rr","areabar","areamax"]
    df.to_csv("ml/savefreq_10m_23_"+str(ii)+".csv",index =None)
    print("save done")
    print(df.tail())
    print(df.shape)     
    
    
    
    

    
    

    
def cal_dif(x):
    max_dif = np.zeros(36)
    min_dif = np.zeros(36)
    # width = np.zeros(10)
    m = np.zeros(36)
    for i in range(0,36):
        temp = x[1000*i:1000*(i+1)]
        # width[i] = np.max(temp)-np.min(temp)
        m[i] = np.mean(temp)
    mm = np.mean(m)
    # avg_width = np.mean(width)
    for i in range(0,36):
        temp = x[1000*i:1000*(i+1)]
        max_dif[i] = np.max(temp) - mm
        min_dif[i] = np.min(temp) - mm
        np.min(temp)
    a = np.max(max_dif[4:-4])
    # /avg_width
    b = np.min(min_dif[4:-4])
    return round(a,3),round(b,3),round(mm,3)
        
def simularity(data):
    list = []
    ind = max_ind(data)
    ref = data[ind,:,0] 
    for i in range(0,23):
        temp = data[i,:,0]    
        list.append(cal(np.squeeze(ref)+2,np.squeeze(temp)+2,ref.shape[0]))
    list.sort()
    print(list)
    
def max_ind(data):
    res =[]
    for j in range(0,data.shape[0]):
        nums = []
        
        for i in range(0,23):
            temp = data[i,:,0]
            nums.append(np.max(temp)-np.min(temp))
            
        max_num_index_list = map(nums.index, heapq.nlargest(1, nums))

        ll = list(max_num_index_list)
    return ll
    
    

    

def plot_fig():   
    plot_err(1,list(range(100)))

def main():
    s1 = timeit.default_timer()  


    find_err()
    pack_ploterr()
    
    
    # for ii in range(1,14):
        # save_freq(ii)
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
if __name__ == '__main__':  

    main()
