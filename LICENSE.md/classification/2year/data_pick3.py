
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
from random import shuffle
import collections
from scipy import signal 
import os
import sys
start = timeit.default_timer()


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
    
    
def rm(X,y):
    """
    THIS FUNCTION REMOVES THE PLANNED EVENTS FROM THE EVENT DATASET
    """
    # X = X.values
    # y= y.values
    X_new=[]
    y_new=[]
    for j in range(0,X.shape[0]):

        if "Planned" in y[j][1]:
            y_new.append(0)
            X_new.append(X[j,:,:,:])
        elif "Trip" in y[j][1]:
            y_new.append(1)
            X_new.append(X[j,:,:,:])
        elif "Lightning" in y[j][1]:
            y_new.append(2)
            X_new.append(X[j,:,:,:])
        elif "Equipment" in y[j][1]:
            y_new.append(3)    
            X_new.append(X[j,:,:,:])
        elif "Weather" in y[j][1]:
            y_new.append(4)       
            X_new.append(X[j,:,:,:])
        elif "Tree" in y[j][1]:
            y_new.append(5)    
            X_new.append(X[j,:,:,:])
        elif "Contamination" in y[j][1]:
            y_new.append(6)
            X_new.append(X[j,:,:,:])
        elif "Proximity" in y[j][1]:
            y_new.append(7)    
            X_new.append(X[j,:,:,:])
        elif "Fire" in y[j][1]:
            y_new.append(8)     
            X_new.append(X[j,:,:,:])            
        elif "Ice" in y[j][1]:
            y_new.append(9)
            X_new.append(X[j,:,:,:])
        elif "RAS" in y[j][1]:
            y_new.append(10)
            X_new.append(X[j,:,:,:])
        elif "Wind" in y[j][1]:
            y_new.append(11)
            X_new.append(X[j,:,:,:])            
        elif "Animal" in y[j][1]:
            y_new.append(12)       
            X_new.append(X[j,:,:,:])

    return  np.array(X_new), np.array(y_new)   




def rd_rof(k):
    path1 = '../pickleset2/'
    list = ['rocof','vp_m','ip_m']

    p1 = open(path1 +'X_S'+str(k)+'_'+str(list[0])+'_6.pickle',"rb")

    pk3  = pd.read_csv(path1 +'y_S'+str(k)+'_'+str(list[2])+'_6.csv')
   
    pk1 = pickle.load(p1)

    X_train = pk1
    y_train = pk3.values
    
    # X_train = X_train[:1]
    # y_train = y_train[:1]
    # print(" X_train.shape",X_train.shape)
    # print("Before y_train.shape",y_train.shape)
    X_train, y_train  = rm3(X_train, y_train)
    # print("X_train.shape",X_train.shape)
    # print("After y_train.shape",y_train.shape)
    return X_train, y_train    
    
def rd_vpm(k):
    path1 = '../pickleset2/'
    list = ['rocof','vp_m','ip_m']

    p1 = open(path1 +'X_S'+str(k)+'_'+str(list[1])+'_6.pickle',"rb")

    pk3  = pd.read_csv(path1 +'y_S'+str(k)+'_'+str(list[2])+'_6.csv')
   
    pk1 = pickle.load(p1)

    X_train = pk1
    y_train = pk3.values
    
    # X_train = X_train[:1]
    # y_train = y_train[:1]
    # print(" X_train.shape",X_train.shape)
    # print("Before y_train.shape",y_train.shape)
    X_train, y_train  = rm3(X_train, y_train)
    # print("X_train.shape",X_train.shape)
    # print("After y_train.shape",y_train.shape)
    return X_train, y_train    
    
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
   
def diff(listA,listB):
    #求交集的两种方式
    retA = [i for i in listA if i in listB]          
    if retA:
        return True 
    else:
        return False    

def get_split(ii):        
    X,y = rd_vpm(ii)
    # print(X.shape)
    # print(y.shape)
    spliter(ii,y)

def spliter(num,y3):        
    a = np.arange(0,y3.shape[0])
    tr,val = train_test_split(a,test_size=0.2)   
    print(tr.shape)
    print(val.shape)
    path2 = 'index5/'
    np.save(path2+'tr_'+str(num)+'.npy',tr) 
    np.save(path2+'val_'+str(num)+'.npy',val)

def mid_data(k,f):

    path2 = 'index5/'
    tr = np.load(path2+'tr_'+str(k)+'.npy') 
    val = np.load(path2+'val_'+str(k)+'.npy') 
    list1 = tr.astype(int).tolist()
    list2 = val.astype(int).tolist()
    if(f ==0):
        X,y = rd_rof(k)
    if(f ==1):
        X,y = rd_vpm(k)    
    X_train = X[list1]
    y_train = y[list1]
    X_val = X[list2]
    y_val = y[list2]
    return X_train,y_train,X_val,y_val
    
def dataPack(f,ii):
    X_train,y_train,X_val,y_val = mid_data(ii,f)
    # for k in range(2,14):
        # X_train2,y_train2,X_val2,y_val2 = mid_data(k,f)
        # X_train = np.concatenate((X_train, X_train2), axis=0)
        # y_train = np.concatenate((y_train, y_train2), axis=0)
        
        # X_val = np.concatenate((X_val, X_val2), axis=0)
        # y_val = np.concatenate((y_val, y_val2), axis=0)       

    return X_train,y_train,X_val,y_val
    
def select(X,y):
    res =[]
    for j in range(0,X.shape[0]):
        nums = []
        
        for i in range(0,23):
            temp = X[j,i,:,0]
            nums.append(np.max(temp)-np.min(temp))
            
        max_num_index_list = map(nums.index, heapq.nlargest(3, nums))
        # min_list = map(nums.index, heapq.nsmallest(3, nums))
        # if(y[j]==3):
            # ll = list(min_list)
        # else:
        ll = list(max_num_index_list)
        # temp = X[j,ll,:,:]
        res.append(ll)
    
    return np.array(res)
    
def cal4(X_train, X_train2, y_train,fname):  

    print("y_train.shape",y_train.shape)
    ar_x1 = np.zeros(X_train.shape[0]) 
    ar_x2 = np.zeros(X_train.shape[0]) 

    for j in range(0,y_train.shape[0]):
        point = np.zeros(90) 
        point2 = np.zeros(90) 
        y = y_train[j]
        vpmean = np.mean(X_train[j,:,:,:])
        
        for i in range(0,3):
            
            x = X_train[j,i,:,:]
            x2 = X_train2[j,i,:,:]
            x= np.squeeze(x)
            x2= np.squeeze(x2)
            
            # print(x.shape)
            zz= Modified_Z(x)
            z2= Modified_Z(x2)
            # print("zz.shape",zz.shape)
            # x2 = np.abs(x2)
            ind = np.argmin(zz)
            ind2 = np.argmin(z2)

            list2 = []
            list3 = []

            list2.append(ind)   
            list3.append(ind2) 

            score = []
            for k in range (0, 89):
                list1 = range(k*120,(k+1)*120)
                if (diff(list1,list2)):
                    point[k] +=1

                    
                if (diff(list1,list3)):
                    point2[k] +=1
                  
        p1 =0    
        ind =0
        for k in range (1, 89):    
            if(p1<point[k]):
                p1 = point[k]
                ind = k
        ar_x1[j] = ind*120
        
        p2=0    
        ind2 =0
        for k in range (1, 89):    
            if(p2<point2[k]):
                p2 = point2[k]
                ind2 = k
        ar_x2[j] = ind2*120
            
    np.save(fname+'_x1.npy',ar_x1) 
    np.save(fname+'_x2.npy',ar_x2)            

    # print("save done")    
    
    


def selected(ii):
    X_train,y_train,X_val,y_val  = dataPack(1,ii)
    X_train2,y_train2,X_val2,y_val2 = dataPack(0,ii)
    index = select(X_train,y_train)

    res1 =[]
    res2 =[]
    for j in range(0,y_train.shape[0]):
        ind = np.squeeze(index[j])
        
        temp1 = X_train[j,ind,:,:]
        temp2 = X_train2[j,ind,:,:]
        res1.append(temp1)
        res2.append(temp2)

    res1 = np.array(res1)
    res2 = np.array(res2)
    
    res3 =[]
    res4 =[]
    index = select(X_val,y_val)
    for j in range(0,y_val.shape[0]):
        ind = np.squeeze(index[j])
        
        temp1 = X_val[j,ind,:,:]
        temp2 = X_val2[j,ind,:,:]
        res3.append(temp1)
        res4.append(temp2)

    res3 = np.array(res3)
    res4 = np.array(res4)    
    
    # print(res1.shape)
    # print(res2.shape)
    return res1,res2,y_train,res3,res4,y_val
    
    
    
def unselected(ii):
    X_train,y_train,X_val,y_val  = dataPack(1,ii)
    X_train2,y_train2,X_val2,y_val2 = dataPack(0,ii)

    return X_train,X_train2,y_train,X_val,X_val2,y_val
    
def get2sec(ii):
    # X_train,y_train,X_val,y_val  = dataPack(0,ii)
    # X_train2,y_train2,X_val2,y_val2 = dataPack(1,ii)
    
    X_train,X_train2,y_train,X_val,X_val2,y_val = selected(ii)
    print(X_train.shape)
    print(X_train2.shape)    
    print(y_train.shape) 
    
    
    
    print(X_val.shape)
    print(X_val2.shape)    
    print(y_val.shape) 
    ff = "gen/set_tr"+str(ii)
    cal4(X_train,X_train2,y_train,ff)
    ff = "gen/set_val"+str(ii)
    cal4(X_val,X_val2,y_val,ff) 

    # return X_train,y_train,X_val,y_val
    
def save_data(ii):

    path3 = "debug/"

    X_train,X_train2,y_train,X_val,X_val2,y_val = selected(ii)

    ff1 = "gen/set_tr"+str(ii)
    ff2 = "gen/set_val"+str(ii)
    vp,rof,label,index = fil_Data(X_train, y_train,X_train2, y_train,ff1)
    vp2,rof2,label2,index2 = fil_Data(X_val,y_val,X_val2,y_val,ff2)    
    
    data = (vp,rof) 
    label = pd.DataFrame(label)
    label.columns = ['label']
    label['index'] = index
    label.to_csv(path3+"index_"+str(ii)+".csv",index = None)
    pickle_out = open(path3+'tr_'+str(ii)+'.pickle',"wb")
    pickle.dump(data, pickle_out, protocol=2)
    pickle_out.close()             

    data = (vp2,rof2) 
    label2 = pd.DataFrame(label2)
    label2.columns = ['label']
    label2['index'] = index2
    label2.to_csv(path3+"index2_"+str(ii)+".csv",index = None)
    pickle_out = open(path3+'val_'+str(ii)+'.pickle',"wb")
    pickle.dump(data, pickle_out, protocol=2)
    pickle_out.close()           

    print("data No. "+str(ii)+" save done")

    
def save_data2(ii):

    path3 = "debug2/"
    # X_train,y_train,X_val,y_val  = dataPack(1,ii)
    # X_train2,y_train2,X_val2,y_val2 = dataPack(0,ii) 
    X_train,X_train2,y_train,X_val,X_val2,y_val = unselected(ii)

    ff1 = "gen/set_tr"+str(ii)
    ff2 = "gen/set_val"+str(ii)
    vp,rof,label,index = fil_Data2(X_train, y_train,X_train2, y_train,ff1)
    vp2,rof2,label2,index2 = fil_Data2(X_val,y_val,X_val2,y_val,ff2)    
    
    data = (vp,rof) 
    label = pd.DataFrame(label)
    label.columns = ['label']
    label['index'] = index
    label.to_csv(path3+"index_"+str(ii)+".csv",index = None)
    pickle_out = open(path3+'tr_'+str(ii)+'.pickle',"wb")
    pickle.dump(data, pickle_out, protocol=2)
    pickle_out.close()             

    data = (vp2,rof2) 
    label2 = pd.DataFrame(label2)
    label2.columns = ['label']
    label2['index'] = index2
    label2.to_csv(path3+"index2_"+str(ii)+".csv",index = None)
    pickle_out = open(path3+'val_'+str(ii)+'.pickle',"wb")
    pickle.dump(data, pickle_out, protocol=2)
    pickle_out.close()           

    print("data No. "+str(ii)+" save done")
    
def load_data(ii):   
    path3 = "debug/"
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
    
    
    
def load_data2(ii):   
    pickle_in = open("gen/X2_"+str(ii)+".pickle","rb")
    X_train,y_train,X_val,y_val= pickle.load(pickle_in)
  
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    return X_train,y_train,X_val,y_val

def fil_Data2(X_train, y_train,X_train2, y_train2,ff): 
    fname1 = ff+"_x1.npy"
    a = np.load(fname1) 
    # print(a)
    fname2 = ff+"_x2.npy"
    b = np.load(fname2)     
    print("X_train.shape",X_train.shape)
    sum1 =0
    # sum2 =0
    if((y_train==y_train2).all()):
        vp = np.zeros((1,23,10800,1))
        rof = np.zeros((1,23,10800,1))
        label = []
        
        index = []
        st1 = []
        st2 = []
        for j in range(0,X_train.shape[0]):
            c= int(a[j])
            d= int(b[j])
            
            if(c+120<10800 and c>-1 and d+120<10800 and d>-1):
            
                y = y_train[j]
                st1 = [] 
                st2 = []
                # vv = np.mean(X_train[j,:,:,:])
                tdata1=[]
                tdata2=[]
                tindex=0
                for i in range(0,23):
                    x = X_train[j,i,:,:]
                    x2 = X_train2[j,i,:,:]
                    temp1 =0
                    temp2 = 0
                    Hlight = range(c,c+120) # vp_m
                    Hlight2 = range(d,d+120)# rocof
                    
                    if(y ==0):
                        temp1 = x[Hlight]
                        temp2 = x2[Hlight]
                        tindex = Hlight[0]
                    elif(y ==2 ):
                        temp1 = x[Hlight2]
                        temp2 = x2[Hlight2]
                        tindex = Hlight[0]
                    elif(y ==1 ):
                        temp1 = x[Hlight2]
                        temp2 = x2[Hlight2]
                        tindex = Hlight[0]
                    elif(y ==3 ):
                        tindex = -1
                        temp1 = x[Hlight2]
                        temp2 = x2[Hlight2]                       
                        
                    temp1 = np.squeeze(temp1)
                    temp2 = np.squeeze(temp2)
                    flag = 0
                        
                    if(temp1[0]!=temp1[-1] and temp2[0]!=temp2[-1]):
                        flag =1

                    else:
                        if(np.max(temp1)!=np.min(temp1) and np.max(temp2)!=np.min(temp2)):
                            flag =1
                            
                    if(flag == 1):
                        tdata1.append(x)
                        tdata2.append(x2)
                    else:
                        tdata1 =[]
                        tdata2 =[]  
                        
                tdata1 = np.array(tdata1) 
                tdata2 = np.array(tdata2) 
  

                if(tdata1.shape[0] == 23 and tdata1.shape[0] == 23 ):
                
                    tdata1 = tdata1.reshape(-1,23,10800,1)
                    tdata2 = tdata2.reshape(-1,23,10800,1)      
                    
                    sum1+=1
                    vp = np.concatenate((vp, tdata1), axis=0)
                    rof = np.concatenate((rof, tdata2), axis=0)

                    index.append(tindex)
                    label.append(y)

      
        index = np.array(index)   
        label = np.array(label)  
        
        vp = vp[1:]

        rof = rof[1:]
 
        print(sum1)
    
        
    return vp,rof,label,index
  



    
def fil_Data(X_train, y_train,X_train2, y_train2,ff): 
    fname1 = ff+"_x1.npy"
    a = np.load(fname1) 
    # print(a)
    fname2 = ff+"_x2.npy"
    b = np.load(fname2)     

    sum1 =0
    # sum2 =0
    if((y_train==y_train2).all()):
        vp = np.zeros((1,3,10800,1))
        rof = np.zeros((1,3,10800,1))
        label = []
        
        index = []
        st1 = []
        st2 = []
        for j in range(0,X_train.shape[0]):
            c= int(a[j])
            d= int(b[j])
            
            if(c+120<10800 and c>-1 and d+120<10800 and d>-1):
            
                y = y_train[j]
                st1 = [] 
                st2 = []
                # vv = np.mean(X_train[j,:,:,:])
                tdata1=[]
                tdata2=[]
                tindex=0
                for i in range(0,3):
                    x = X_train[j,i,:,:]
                    x2 = X_train2[j,i,:,:]
                    temp1 =0
                    temp2 = 0
                    Hlight = range(c,c+120) # vp_m
                    Hlight2 = range(d,d+120)# rocof
                    
                    if(y ==0):
                        temp1 = x[Hlight]
                        temp2 = x2[Hlight]
                        tindex = Hlight[0]
                    elif(y ==2 ):
                        temp1 = x[Hlight2]
                        temp2 = x2[Hlight2]
                        tindex = Hlight2[0]
                    elif(y ==1 ):
                        temp1 = x[Hlight]
                        temp2 = x2[Hlight]
                        tindex = Hlight[0]
                    elif(y ==3 ):
                        tindex = -1
                        temp1 = x[Hlight2]
                        temp2 = x2[Hlight2]                       
                        
                    temp1 = np.squeeze(temp1)
                    temp2 = np.squeeze(temp2)
                    flag = 0
                        
                    if(temp1[0]!=temp1[-1] and temp2[0]!=temp2[-1]):
                        flag =1

                    else:
                        if(np.max(temp1)!=np.min(temp1) and np.max(temp2)!=np.min(temp2)):
                            flag =1
                            
                    if(flag == 1):
                        tdata1.append(x)
                        tdata2.append(x2)
                    else:
                        tdata1 =[]
                        tdata2 =[]  
                        
                tdata1 = np.array(tdata1) 
                tdata2 = np.array(tdata2) 
  

                if(tdata1.shape[0] == 3 and tdata1.shape[0] == 3 ):
                
                    tdata1 = tdata1.reshape(-1,3,10800,1)
                    tdata2 = tdata2.reshape(-1,3,10800,1)     
                    
                    sum1+=1
                    vp = np.concatenate((vp, tdata1), axis=0)
                    rof = np.concatenate((rof, tdata2), axis=0)

                    index.append(tindex)
                    label.append(y)

      
        index = np.array(index)   
        label = np.array(label)  
        
        vp = vp[1:]

        rof = rof[1:]
 
        print(sum1)
    
        
    return vp,rof,label,index
  

def save_mtf(ii):   
    vp,rof,df1,vp2,rof2,df2 = load_data(ii)
    mtf = MarkovTransitionField(image_size=120,strategy='quantile',n_bins=10)
    
    
    aa = np.zeros((1,120,120,3))
    bb = np.zeros((1,120,120,3))
    label = []
    for j in range(vp.shape[0]):
        y = df1["label"].values[j]
        ind = df1["index"].values[j]
        st1 = [] 
        st2 = []        
        
        for i in range(0,3):
            x = vp[j,i,:,:]
            x2 = rof[j,i,:,:]
            
            if(y !=3):
                x = x[ind:ind+120]
                x2 = x2[ind:ind+120]
                
            flag = 0
            temp1 = x
            temp2 = x2
            if(temp1[0]!=temp1[-1] and temp2[0]!=temp2[-1]):
                flag =1

            else:
                if(np.max(temp1)!=np.min(temp1) and np.max(temp2)!=np.min(temp2)):
                    flag =1                
            if(flag == 1):    
                temp1 = mtf.fit_transform(x.reshape(-1,120))
                temp2 = mtf.fit_transform(x2.reshape(-1,120))
                st1.append(temp1[0])
                st2.append(temp2[0])
        st1 =  np.array(st1)
        st2 =  np.array(st2)
        if(st1.shape[0] == 3 and st2.shape[0] == 3 ):
            st1 = st1.reshape(-1,120,120,3)
            st2 = st2.reshape(-1,120,120,3)   
        
            aa = np.concatenate((aa, st1), axis=0)
            bb = np.concatenate((bb, st2), axis=0)        
            label.append(y)
    y_train = np.array(label)   
    tr_vp = aa[1:]
    tr_rof = bb[1:]



    aa = np.zeros((1,120,120,3))
    bb = np.zeros((1,120,120,3))
    label = []
    for j in range(vp2.shape[0]):
        y = df2["label"].values[j]
        ind = df2["index"].values[j]
        st1 = [] 
        st2 = []        
        
        for i in range(0,3):
            x = vp[j,i,:,:]
            x2 = rof[j,i,:,:]
            
            if(y !=3):
                x = x[ind:ind+120]
                x2 = x2[ind:ind+120]
                
            flag = 0
            temp1 = x
            temp2 = x2
            if(temp1[0]!=temp1[-1] and temp2[0]!=temp2[-1]):
                flag =1

            else:
                if(np.max(temp1)!=np.min(temp1) and np.max(temp2)!=np.min(temp2)):
                    flag =1                
            if(flag == 1):    
                temp1 = mtf.fit_transform(x.reshape(-1,120))
                temp2 = mtf.fit_transform(x2.reshape(-1,120))
                st1.append(temp1[0])
                st2.append(temp2[0])
        st1 =  np.array(st1)
        st2 =  np.array(st2)
        if(st1.shape[0] == 3 and st2.shape[0] == 3 ):
            st1 = st1.reshape(-1,120,120,3)
            st2 = st2.reshape(-1,120,120,3)   
        
            aa = np.concatenate((aa, st1), axis=0)
            bb = np.concatenate((bb, st2), axis=0)        
            label.append(y)
    y_val = np.array(label)   
    val_vp = aa[1:]
    val_rof = bb[1:]
        
        
    print("val_vp",val_vp.shape)
    print("val_rof",val_rof.shape)
    print("y_val",y_val.shape)
    print("tr_vp.shape",tr_vp.shape)
    print("tr_rof.shape",tr_rof.shape)       
    print("y_train",y_train.shape)        
    
    X_train = np.concatenate((tr_vp, tr_vp), axis=3)
    X_val = np.concatenate((val_vp, val_rof), axis=3)
    data = (X_train,y_train,X_val,y_val) 
  
    pickle_out = open('gen/X2_'+str(ii)+'.pickle',"wb")
    pickle.dump(data, pickle_out, protocol=2)
    pickle_out.close()

    print("MTF saved of No. "+str(ii))
    
def main():
    s1 = timeit.default_timer()  
    # rd2(1,1)
    for ii in range(7,14):
        # get_split(ii)
        # get2sec(ii)
        # save_data(ii)
        # load_data(ii)
        save_mtf(ii)
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
if __name__ == '__main__':  
    main()
