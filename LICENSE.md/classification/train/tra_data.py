
# This is to track the training and testing data
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
        ww = temp[0]+"_"+temp[1]+"_"+temp[2]+"_"+temp[3]
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
    # ,np.array(word)

    


def rd2(k,i):
    path1 = '../pickleset2/'

    # list = ['ip_m','vp_a','va_m','va_a','vb_m','vb_a','vc_m','vc_a','rocof']
    list = ['rocof','vp_m','ip_m']

    p1 = open(path1 +'X_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")

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
        z_score =  dev_med/(1.253314*mad)
        
    return z_score
   
def diff(listA,listB):
    #求交集的两种方式
    retA = [i for i in listA if i in listB]          
    if retA:
        return True 
    else:
        return False    

def get_split(ii):        
    X,y = rd2(ii,0)
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
    X,y = rd2(k,f)
    
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
            
            x = X_train[j,i,:,:]/vpmean
            x2 = X_train2[j,i,:,:]
            x= np.squeeze(x)
            x2= np.squeeze(x2)
            
            # print(x.shape)
            zz= Modified_Z(x)
            z2= Modified_Z(x2)
            # print("zz.shape",zz.shape)
            x2 = np.abs(x2)
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

    print("save done")    
    
    


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
    

def save_data(ii):
    X_train,y_train,X_val,y_val  = dataPack(1,ii)
    X_train2,y_train2,X_val2,y_val2 = dataPack(0,ii) 
    ff1 = "gen/set_tr"+str(ii)
    ff2 = "gen/set_val"+str(ii)
    # vp,rof,label = fil_Data(X_train, y_train,X_train2, y_train2,ff1)
    vp2,rof2,label2,index = fil_Data(X_val,y_val,X_val2,y_val2,ff2)
    
    # print("vp2.shape",vp2.shape)
    # print("rof2.shape",rof2.shape)
    # print("label2.shape",label2.shape)
    # print("index.shape",index.shape)
    
    return vp2,rof2,label2,index
    
def pack():
    vp,rof,label,index = save_data(1)
    for i in range(2,14):
        a,b,lab,ind = save_data(i)
        vp = np.concatenate((vp, a), axis=0)
        rof = np.concatenate((rof, b), axis=0)
        label = np.concatenate((label, lab), axis=0)
        index = np.concatenate((index, ind), axis=0)
        
    data = (vp,rof,label,index) 
    label = pd.DataFrame(label)
    label.to_csv("cc.csv",index = None)
    pickle_out = open('gen/track_data.pickle',"wb")
    pickle.dump(data, pickle_out, protocol=2)
    pickle_out.close()         
    
def load_data(ii):   
    pickle_in = open("gen/X2_"+str(ii)+".pickle","rb")
    X_train,y_train,X_val,y_val= pickle.load(pickle_in)
  
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    return X_train,y_train,X_val,y_val
    
def fil_Data(X_train, y_train,X_train2, y_train2,ff): 
    fname1 = ff+"_x1.npy"
    a = np.load(fname1) 
    # print(a)
    fname2 = ff+"_x2.npy"
    b = np.load(fname2)     

    sum1 =0
    # sum2 =0
    if((y_train==y_train2).all()):
        vp = np.zeros((1,10800,3))
        rof = np.zeros((1,10800,3))
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
  

                if(tdata1.shape[0] == 3 and tdata1.shape[0] == 3 ):
                
                    tdata1 = tdata1.reshape(-1,10800,3)
                    tdata2 = tdata2.reshape(-1,10800,3)        
                    
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
  
    
def main():
    s1 = timeit.default_timer()  
    pack()
    
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
if __name__ == '__main__':  
    main()
