
# This is for Feq CWT^2 use
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import pandas as pd
import pywt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import heapq
from sklearn.model_selection import learning_curve, GridSearchCV  
from sklearn.svm import SVC    
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
import pywt
import pickle
import timeit
import datetime
from pyts.image import MarkovTransitionField

from numpy import array, sign, zeros
from scipy.interpolate import interp1d
import scipy.signal

from scipy.stats import pearsonr,spearmanr,kendalltau
from scipy import signal 
import os
import sys
from statistics import variance 


import datapick


start = timeit.default_timer()
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])






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
    


# 23 PMU separate:    
def plt_2(): 

    
    vp,rof,df1,vp2,rof2,df2 = datapick.load_data(1)
    label = df1['label'].values
    index = df1['index'].values
    for j in range(0,1):
        plt.figure( figsize=(15,8))
        for i in range(0,3):
            
            x = vp[j,:,i]
            x2 = rof[j,:,i]

            x= np.squeeze(x)
            x2= np.squeeze(x2)     
            
            zz= Modified_Z(x)
            z2= Modified_Z(x2)       


            ind = np.argmin(zz)
            ind2 = np.argmin(z2)
            

            plt.subplot(2,1,1)
            
            plt.scatter(ind,x[ind],marker='p',c='',edgecolors='r',zorder=10)
            plt.plot(range(len(x)),x,zorder=0)
            plt.ylabel("Vp_m ")
            if(index[j]>0):
                Hlight = range(index[j],index[j]+120)
            else: 
                Hlight = 0
            
            if Hlight!=0:
                plt.scatter(Hlight,x[Hlight], color=lighten_color('g', 0.2), linewidth=22.0, linestyle="-",zorder=-1)
                   
            plt.subplot(2,1,2)
            plt.plot(range(len(x2)),x2,zorder=0)
            plt.scatter(ind2,x2[ind2],marker='p',c='',edgecolors='r',zorder=10)
            plt.ylabel("ROCOF")
            if Hlight!=0:
                plt.plot(Hlight,x2[Hlight], color=lighten_color('g', 0.2), linewidth=22.0, linestyle="-",zorder=-1)

        plt.title('label = '+str(label[j])) 

        plt.tight_layout() 
        plt.savefig("gen/tra_"+str(j))   


    print("save done")
    
def load_data(ii):   
    path3 = "debug2/"
    pickle_in = open(path3+"tr_"+str(ii)+".pickle","rb")
    vp,rof = pickle.load(pickle_in)

    pickle_in = open(path3+"val_"+str(ii)+".pickle","rb")
    vp2,rof2 = pickle.load(pickle_in)

    df1 = pd.read_csv(path3+"index_"+str(ii)+".csv")
    df2 = pd.read_csv(path3+"index2_"+str(ii)+".csv")
    
    print("train vp shape",vp.shape)
    print("train rof shape",rof.shape)
    print("train vp2 shape",vp2.shape)
    print("train rof2 shape",rof2.shape)
    
    print("df1 shape",df1.shape)
    print("df2 shape",df2.shape)    
    return vp,rof,df1,vp2,rof2,df2
    
    


    
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
        # print("len(n1)",len(n1))
        # print("len(n2)",len(n2))
        # print(y1[ind-k2:ind].shape[0])
        # print(y1[ind:k3].shape[0])
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
    X_train, y_train  = datapick.rd_rof(ii)
    X_train2, _  = datapick.rd_vpm(ii)
    
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
            x2= np.squeeze(X_train2[j,i,:,:])
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
    duan = np.zeros(30)
    for j in range(0,len(list)):
        for i in range(0,30):
            if list[j]<360*(i+1) and list[j]>360*(i):
                duan[i]+=1
    # print(duan)
        
    list_a = duan.tolist()
    res = list_a.index(max(list_a))
    point = np.median(np.array(list))
    # print(res)
    return res,int(point)    
    
def compare_mead(list1,list2):
    t1 = np.array(list1)
    r1 = np.median(t1)
    
    t2 = np.array(list2)
    r2 = np.median(t2)
    res = 0
    temp = np.abs(r1-r2)
    if(temp<=10*60 and r1>2*1080 and r2>2*1080 and r1<8*1080 and r2<8*1080 ):
        return 1
    else:
        return 0
    # return res
    
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
    X_train, y_train  = datapick.rd_rof(ii)
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
        kk = np.zeros(12)
        # y = df1["label"].values[j]
        y = y_train[j]



        ref,ll = choose(X_train[j,:,:,:])
        # ref = ref[3600-120:3600+180]
        # print("ref.shape",ref.shape)
        for i in range(0,23):
            x = np.squeeze(X_train[j,i,:,:])
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
            
            if(point>180 and point<10800-180):
                temp = x[point-180:point+180]
            elif(point>10800-180):
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
    df.to_csv("ml/savefreq_23_"+str(ii)+".csv",index =None)
    print("save done")
    print(df.tail())
    print(df.shape)     
    
    
    
    
def save_23(ii):

    X_train, y_train  = datapick.rd_ipm(ii)
    
    # X_train, y_train  = rd_f(2)
    print(X_train.shape)
    print(y_train.shape)
    # X_train.shape[0]
    list = []
    list2 = []
    list3 = []
    for j in range(0,X_train.shape[0]):
        if(j%10==0):
            print("j=",j)

        # y = df1["label"].values[j]
        y = y_train[j]


        # nor = np.mean(X_train[j,:,:,:])

        for i in range(0,23):
            x = np.squeeze(X_train[j,i,:,:])

            cc = x
            # list.append(np.max(cc) - np.min(cc))
            plt.plot(range(len(cc)),cc,zorder=0)
            temp1 = cal(cc)
            temp2 = cal_D(temp1)
            temp3 = envelop(cc)
            list.append(y)
            list2.append(temp2)
            list3.append(temp3)

        # plt.title('label = '+str(y)) 
        # plt.subplot(4,6,24)
        # plt.title('avg = delta'+str(round(np.mean(list),6))+"; max = "+str(round(np.max(list),6))) 
        # s1 = round(np.mean(list2),2)
        # s2 = round(np.mean(list3),2)
        
        # st.append(s1)
        # st.append(s2)
        # st.append(y)
        
    # st = np.array(list,list2,list3)
    # st = st.reshape((-1,3))
    df = pd.DataFrame({"F1":list2,"F2":list3,"label":list})
    df.to_csv("save_23_"+str(ii)+".csv",index =None)
    print("save done")
    print(df.tail())
    print(df.shape)        



    
def envelop(y):
    
    list1 =[]
    ind1 = []
    list2 =[]
    ind2 =[]
    for i in range(1,len(y)-1):
        if(y[i-1]<=y[i] and y[i]>=y[i+1]):
            list1.append(y[i])
            ind1.append(i)
        if(y[i-1]>=y[i] and y[i]<=y[i+1]):
            list2.append(y[i])
            ind2.append(i)
    # print(ind1)
    # print(ind2)
    ind1 = np.array(ind1)
    ind2 = np.array(ind2)
    
    
    res = (variance(y[ind1])*len(ind1)/variance(y))**(0.5)+(variance(y[ind2])*len(ind2)/variance(y))**(0.5)
    # print(k) #refer to F2
    return round(res,2)
    
def cal(y):
    list = [] #Refer to B
    for i in range(0,y.shape[0]-1):
        list.append(y[i+1]-y[i])
    # print(len(list))
    for i in range(len(list)):
        if(list[i]>=0):
            list[i] = 1
        else:
            list[i] = -1
    list2 =[] #Refer to C
    cnt =1 
    for i in range(len(list)-1):
        if(list[i]==1 and list[i+1]==1):
            cnt+=1
        elif(list[i]==-1 and list[i+1]==-1):
            cnt+=1
        else:
            
            list2.append(cnt)
            cnt = 1
    list2.append(cnt)
    # print("last cnt",cnt)
    # print(list)
    # print(list2)
    list2 = np.array(list2)
    return list2
    
def cal_D(arr):
    n = np.zeros(len(arr)) #Refer to D
    m=2
    for i in range(0,int(arr.shape[0]/2),2): 
        if(arr[i] == arr[i+m] and arr[i+1] == arr[i+1+m]):
            n[i] =1
            n[i+m] =1
            n[i+1] =1
            n[i+1+m] =1
    sum = 0
    for i in range(len(arr)):
        sum+= arr[i]*n[i]
    # print(n)
    # print(sum)     
    return sum
    
def rd_f(k):
    path1 = '../pickleset3/'
    list = ['rocof','vp_m','ip_m','f']

    p1 = open(path1 +'X_S'+str(k)+'_'+str(list[3])+'_6.pickle',"rb")

    pk3  = pd.read_csv(path1 +'y_S'+str(k)+'_'+str(list[3])+'_6.csv')
   
    pk1 = pickle.load(p1)

    X_train = pk1
    y_train = pk3.values

    X_train, y_train  = datapick.rm3(X_train, y_train)

    return X_train, y_train        
    
    
    
def plot_osc(ii):
    # ii = 1
    # vp,rof,df1,vp2,rof2,df2 = load_data(ii)
    X_train, y_train  = datapick.rd_vpm(1)
    
    # X_train, y_train  = rd_f(2)
    print(X_train.shape)
    print(y_train.shape)
    # X_train.shape[0]
    for j in range(0,22):
        list = []
        list2 = []
        list3 = []
        # y = df1["label"].values[j]
        y = y_train[j]
        if(y==0):
            plt.figure( figsize=(20,5))
            nor = np.mean(X_train[j,:,:,:])
            # ref,ll = choose(X_train[j,:,:,:])
            # ref = ref[3600-120:3600+180]
            # print("ref.shape",ref.shape)
            for i in range(0,23):
                x = np.squeeze(X_train[j,i,:,:])/nor
                # print(x.shape)
                # zz = Modified_Z(x)

                plt.subplot(4,6,i+1)
                # cc = x[3600-240:3600+300]
                cc = x
                list.append(np.max(cc) - np.min(cc))
                plt.plot(range(len(cc)),cc,zorder=0)
                temp1 = cal(cc)
                temp2 = cal_D(temp1)
                temp3 = envelop(cc)
                list2.append(temp2)
                list3.append(temp3)
                plt.xlabel("F1: "+str(temp2)+"; F2: "+str(temp3))
                plt.ylabel("vp_m")
            plt.title('label = '+str(y)) 
            plt.subplot(4,6,24)
            plt.title('avg = delta'+str(round(np.mean(list),6))+"; max = "+str(round(np.max(list),6))) 
            s1 = str(round(np.mean(list2),2))
            s2 = str(round(np.mean(list3),2))
            plt.xlabel("Avg F1:"+ s1+"  F2:"+s2 )
            plt.tight_layout()         
            plt.savefig("gen/vpm23_"+str(ii)+"_"+str(j)) 
                
        

def cal_rate(x):
    
    x = x+1
    res = 10*(np.mean(x)-np.min(x))/np.mean(x)
    res2 = 10*np.max(x)/np.mean(x)
    return round(res,2),round(res2,2)
    
    
    #return up, down, and mean
def cal_dif(x):
    max_dif = np.zeros(10)
    min_dif = np.zeros(10)
    # width = np.zeros(10)
    m = np.zeros(10)
    for i in range(0,10):
        temp = x[1080*i:1080*(i+1)]
        # width[i] = np.max(temp)-np.min(temp)
        m[i] = np.mean(temp)
    mm = np.mean(m)
    # avg_width = np.mean(width)
    for i in range(0,10):
        temp = x[1080*i:1080*(i+1)]
        max_dif[i] = np.max(temp) - mm
        min_dif[i] = np.min(temp) - mm
        np.min(temp)
    a = np.max(max_dif[2:-2])
    # /avg_width
    b = np.min(min_dif[2:-2])
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
    
    

    
def plt_septrain(ii): 

    # cnt = 1
    vp,rof,df1,vp2,rof2,df2 = datapick.load_data(ii)
    label = df1['label'].values
    index = df1['index'].values
    for j in range(0,vp2.shape[0]):
        plt.figure( figsize=(15,10))
        if(label[j]==2):
        

        
            if(index[j]>60 and index[j]< 10800-60):
                Hlight = range(index[j]-60,index[j]+60)
            elif(index[j]<60):
                Hlight = range(0,120)
                # x2 = x2[0:120]      

            elif(index[j]>10800-60):
                
                Hlight = range(10800-120,10800)

            for i in range(0,3):
                
                x = vp[j,i,:,:]
                x2 = rof[j,i,:,:]

                x= np.squeeze(x)
                x2= np.squeeze(x2)     
                
                zz= Modified_Z(x)
                z2= Modified_Z(x2)       


                ind = np.argmin(zz)
                ind2 = np.argmin(z2)
                

                plt.subplot(4,3,i+1)
                
                plt.scatter(ind,x[ind],marker='p',c='',edgecolors='r',zorder=10)
                # plt.title("index = "+str(index[j]))
                plt.plot(range(len(x)),x,zorder=0)
                plt.ylabel("Vp_m ")
                plt.xlabel("argmin="+str(ind))
                
                if Hlight!=0:
                    plt.fill_between(Hlight,x[Hlight],np.min(x[Hlight]), color='g',alpha=0.25,zorder=11)
                elif Hlight==0 and i ==1:
                    plt.title("index = "+str(index[j]))
                    
                plt.subplot(4,3,i+4)
                plt.plot(range(len(x2)),x2,zorder=0)
                plt.scatter(ind2,x2[ind2],marker='p',c='',edgecolors='r',zorder=10)
                plt.ylabel("ROCOF")
                plt.xlabel("argmin="+str(ind2))
                if Hlight!=0:
                    # plt.plot(Hlight,x2[Hlight], color=lighten_color('g', 0.2), linewidth=22.0, linestyle="-",zorder=-1)
                    plt.fill_between(Hlight,x2[Hlight],np.min(x2[Hlight])-0.05, color='g',alpha=0.25,zorder=11)
                    
                t1 = x[Hlight]
                t2 = x2[Hlight]
                plt.subplot(4,3,i+7)
                plt.plot(range(t1.shape[0]),t1,zorder=0)
                plt.ylabel("Vp_m")
                plt.xlabel("index = "+str(index[j]))
                
                plt.subplot(4,3,i+10)
                plt.plot(range(t2.shape[0]),t2,zorder=0)
                plt.ylabel("ROCOF")
            
                    
            plt.title('label = '+str(label[j])) 

            plt.tight_layout() 
            plt.savefig("gen/tra_"+str(ii)+"_"+str(j))   
            # cnt = cnt+1

    print("save done")
        
    
    
    
    
def plt_sep(ii): 

    # cnt = 1
    h_band = 180
    vp,rof,df1,vp2,rof2,df2 = datapick.load_data(ii)
    label = df2['label'].values
    index = df2['index'].values
    for j in range(0,vp2.shape[0]):
        plt.figure( figsize=(15,10))
        if(label[j]==0):
            if(index[j]>h_band and index[j]< 10800-h_band):
                Hlight = range(index[j]-h_band,index[j]+h_band)
            elif(index[j]<h_band):
                Hlight = range(0,2*h_band)
                # x2 = x2[0:120]      

            elif(index[j]>10800-h_band):
                
                Hlight = range(10800-2*h_band,10800)
                
                
            for i in range(0,3):
                
                x = vp2[j,i,:,:]
                x2 = rof2[j,i,:,:]

                x= np.squeeze(x)
                x2= np.squeeze(x2)     
                
                zz= Modified_Z(x)
                z2= Modified_Z(x2)       


                ind = np.argmin(zz)
                ind2 = np.argmin(z2)
                

                plt.subplot(4,3,i+1)
                
                plt.scatter(ind,x[ind],marker='p',c='',edgecolors='r',zorder=10)
                # plt.title("index = "+str(index[j]))
                plt.plot(range(len(x)),x,zorder=0)
                plt.ylabel("Vp_m ")
                plt.xlabel("argmin="+str(ind))
                
                if Hlight!=0:
                    plt.fill_between(Hlight,x[Hlight],np.min(x[Hlight]), color='g',alpha=0.25,zorder=11)
                # elif Hlight==0 and i ==1:
                    # plt.title("index = "+str(index[j]))
                    
                    
                plt.subplot(4,3,i+4)
                plt.plot(range(len(x2)),x2,zorder=0)
                plt.scatter(ind2,x2[ind2],marker='p',c='',edgecolors='r',zorder=10)
                plt.ylabel("ROCOF")
                plt.xlabel("argmin="+str(ind2))
                
                if Hlight!=0:
                    # plt.plot(Hlight,x2[Hlight], color=lighten_color('g', 0.2), linewidth=22.0, linestyle="-",zorder=-1)
                    plt.fill_between(Hlight,x2[Hlight],np.min(x2[Hlight])-0.05, color='g',alpha=0.25,zorder=11)
                    
                t1 = x[Hlight]
                t2 = x2[Hlight]
                plt.subplot(4,3,i+7)
                plt.plot(range(t1.shape[0]),t1,zorder=0)
                plt.ylabel("Vp_m")
                plt.xlabel("index = "+str(index[j]))
                
                plt.subplot(4,3,i+10)
                plt.plot(range(t2.shape[0]),t2,zorder=0)
                plt.ylabel("ROCOF")                    
                    
            plt.title('label = '+str(label[j])) 

            plt.tight_layout() 
            plt.savefig("gen/tra_"+str(ii)+"_"+str(j))   
            # cnt = cnt+1

    print("save done")
    

def plot_fig():   
    plot_err(1,list(range(100)))

def main():
    s1 = timeit.default_timer()  

    
    find_err()
    pack_ploterr()
    
    
    # plot_err(12,78," ")
    # for ii in range(1,14):
        # plt_sep(ii)
        # plt_septrain(ii)
        # plot_osc(ii)
        # save_23(ii)
        # save_freq(ii)
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
if __name__ == '__main__':  

    main()
