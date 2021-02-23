
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



def rdy(k,i):
    path1 = '../pickleset2/'

    list = ['rocof','vp_m','ip_m']
    # y_S1_vp_m_6
    pk3  = pd.read_csv(path1 +'y_S'+str(k)+'_'+str(list[i])+'_6.csv')
    # pk3 = np.array(pk3)
    return pk3   


def rd2(k,i):
    path1 = '../pickleset2/'

    # list = ['ip_m','vp_a','va_m','va_a','vb_m','vb_a','vc_m','vc_a','rocof']
    list = ['rocof','vp_m','ip_m']

    p1 = open(path1 +'X_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")

    pk3  = pd.read_csv(path1 +'y_S'+str(k)+'_'+str(list[2])+'_6.csv')
    
    # pk3 = pk3[:,1]   
    # pk3 = pk3.astype(np.int32)      

    pk1 = pickle.load(p1)

    X_train = pk1
    y_train = pk3.values
    X_train, y_train  = rm(X_train, y_train)


    return X_train, y_train    
    
def Modified_Z(data):
    c = 1.4826
    median = np.median(data)
    # print(median)
    # print("median.shape",median.shape)
    dev_med = np.array(data) -median
    # print("dev_med.shape",dev_med.shape)
    mad = np.median(np.abs(dev_med))
    z_score = dev_med/(mad*mad)
    return z_score
   
def diff(listA,listB):
    #求交集的两种方式
    retA = [i for i in listA if i in listB]          
    if retA:
        return True 
    else:
        return False    




def get_split(num,y3):        
    a = np.arange(0,y3.shape[0])
    tr,val = train_test_split(a,test_size=0.2)   
    print(tr.shape)
    print(val.shape)
    path2 = 'index5/'
    np.save(path2+'tr_'+str(num)+'.npy',tr) 
    np.save(path2+'val_'+str(num)+'.npy',val)

        # add split
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
    
def cal4(X_train, y_train,X_train2, y_train2,fname):  
    print(X_train.shape)
    if((y_train==y_train2).all()):
        print("y_train.shape",y_train.shape)
        ar_x1 = np.zeros(X_train.shape[0]) 
        ar_x2 = np.zeros(X_train.shape[0]) 
        
        for j in range(0,X_train.shape[0]):
            point = np.zeros(90) 
            point2 = np.zeros(90) 
            y = y_train[j]
            for i in range(0,X_train.shape[1]):
                
                x = X_train[j,i,:,:]
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
                k = np.mean(x)
                list2 = []
                list3 = []

                list2.append(ind)   
                list3.append(ind2) 

                score = []
                for ii in range (0, 89):
                    list1 = range(ii*120,(ii+1)*120)
                    if (diff(list1,list2)):
                        point[ii] +=1
    
                        
                    if (diff(list1,list3)):
                        point2[ii] +=1
                      
            p1 =0    
            ind =0
            for ii in range (1, 89):    
                if(p1<point[ii]):
                    p1 = point[ii]
                    ind = ii
            ar_x1[j] = ind*120
            
            p2=0    
            ind2 =0
            for ii in range (1, 89):    
                if(p2<point2[ii]):
                    p2 = point2[ii]
                    ind2 = ii
            ar_x2[j] = ind2*120
            
            
            
    np.save(fname+"x1.npy",ar_x1)
    np.save(fname+"x2.npy",ar_x2)
    print("save done")    
    
def get2sec(ii):
    X_train,y_train,X_val,y_val  = dataPack(0,ii)
    X_train2,y_train2,X_val2,y_val2 = dataPack(1,ii)
    ff = "gen/Linetr_"+str(ii)
    cal4(X_train,y_train,X_train2,y_train2,ff)
    ff = "gen/Lineval_"+str(ii)
    cal4(X_val,y_val,X_val2,y_val2,ff) 

    return X_train,y_train,X_val,y_val
    
def save_data(ii):
    X_train,y_train,X_val,y_val  = dataPack(0,ii)
    X_train2,y_train2,X_val2,y_val2 = dataPack(1,ii) 
    ff1 = "gen/Linetr_"+str(ii)
    ff2 = "gen/Lineval_"+str(ii)
    vp,rof,label = fil_Data(X_train, y_train,X_train2, y_train2,ff1)
    vp2,rof2,label2 = fil_Data(X_val,y_val,X_val2,y_val2,ff2)
    
    print("vp.shape",vp.shape)
    print("rof.shape",rof.shape)
    print("label.shape",label.shape)
    # print(y_val.shape)  
    
    X_train = np.concatenate((vp, rof), axis=3)
    X_val = np.concatenate((vp2, rof2), axis=3)
    data = (X_train,label,X_val,label2) 
  
    pickle_out = open('gen/X_'+str(ii)+'.pickle',"wb")
    pickle.dump(data, pickle_out, protocol=2)
    pickle_out.close()     
    
def load_data(ii):   
    pickle_in = open("gen/X_"+str(ii)+".pickle","rb")
    X_train,y_train,X_val,y_val= pickle.load(pickle_in)
  
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_val.shape)
    # print(y_val.shape)
    return X_train,y_train,X_val,y_val
    
def fil_Data(X_train, y_train,X_train2, y_train2,ff): 
    fname1 = ff+"x1.npy"
    a = np.load(fname1) 
    # print(a)
    fname2 = ff+"x2.npy"
    b = np.load(fname2)     

    sum1 =0
    # sum2 =0
    if((y_train==y_train2).all()):
        vp = np.zeros((1,120,120,3))
        rof = np.zeros((1,120,120,3))
        label = []
        for j in range(0,X_train.shape[0]):
            c= int(a[j])
            d= int(b[j])
            
            if(c+120<10800 and c>-1 and d+120<10800 and d>-1):
            
                y = y_train[j]
                st1 = [] 
                st2 = []
                vv = np.mean(X_train[j,:,:,:])
                for i in range(0,3):
                    x = X_train[j,i,:,:]/vv
                    x2 = X_train2[j,i,:,:]
                    
                    Hlight = range(c,c+120) # vp_m
                    Hlight2 = range(d,d+120)# rocof
                    

                    x = x[Hlight]
                    x2 = x2[Hlight2]

                    temp1 = np.squeeze(x)
                    temp2 = np.squeeze(x2)
                    flag = 0
                        
                    if(temp1[0]!=temp1[-1] and temp2[0]!=temp2[-1]):
                        flag =1

                    else:
                        if(np.max(temp1)!=np.min(temp1) and np.max(temp2)!=np.min(temp2)):
                            flag =1
                            
                    if(flag == 1):
                        # uniform  quantile
                        mtf = MarkovTransitionField(image_size=120,strategy='normal',n_bins=10)
                        
                        # print("x.shape",x.shape)
                        # print("x2.shape",x2.shape)
                        temp1 = mtf.fit_transform(x.reshape(-1,120))
                        # mtf2 = MarkovTransitionField(image_size=120,strategy='quantile',n_bins=10)
                        temp2 = mtf.fit_transform(x2.reshape(-1,120))
                        st1.append(temp1[0])
                        st2.append(temp2[0])
                    else:
                        st1 =[]
                        st2 =[]
                        print("Yeah1")
                st1 = np.array(st1) 
                st2 = np.array(st2)     

                if(st1.shape[0] == 3 and st2.shape[0] == 3 ):
                    st1 = st1.reshape(-1,120,120,3)
                    st2 = st1.reshape(-1,120,120,3)
                    # print("Yeah2")
                    sum1+=1
                    vp = np.concatenate((vp, st1), axis=0)
                    rof = np.concatenate((rof, st2), axis=0)
                    label.append(y)

      
        label = np.array(label)   

        vp = vp[1:]

        rof = rof[1:]
 
        print(sum1)
    
        
    return vp,rof,label
  
def select(X):

    res =[]
    for j in range(0,X.shape[0]):
        nums = []
        for i in range(0,23):
            temp = X[j,i,:,0]
            nums.append(np.max(temp)-np.min(temp))
            
        max_num_index_list = map(nums.index, heapq.nlargest(3, nums))
        ll = list(max_num_index_list)
        temp = X[j,ll,:,:]
        res.append(temp)

    return np.array(res)
    
def MTF(temp,yy):

    # print(temp.shape)
    t1 =[]
    t2 =[]
    label = []
    vp = np.zeros((1,100,100,3))
    rof = np.zeros((1,100,100,3))    
    mtf = MarkovTransitionField(image_size=100,strategy='uniform',n_bins=10)
    # temp.shape[0]
    for k in range(0,temp.shape[0]):
        st1 =[]
        st2 =[]
        y = yy[k]
        # temp.shape[1]
        for j in range(0,3):
        

            x = temp[k,j,:,0]
            x2 = temp[k,j,:,1]   


            temp1 = np.squeeze(x)
            temp2 = np.squeeze(x2)
            
            flag = 0
                
            if(temp1[0]!=temp1[-1] and temp2[0]!=temp2[-1]):
                flag =1

            else:
                if(np.max(temp1)!=np.min(temp1) and np.max(temp2)!=np.min(temp2)):
                    flag =1
                    
            if(flag == 1):        

                temp1 = mtf.fit_transform(x.reshape(-1,x.shape[0]))
                # mtf2 = MarkovTransitionField(image_size=120,strategy='quantile',n_bins=10)
                temp2 = mtf.fit_transform(x2.reshape(-1,120))
                st1.append(temp1[0])
                st2.append(temp2[0])
            
            else:
                st1 = []
                st2 =[]
        st1 = np.array(st1) 
        st2 = np.array(st2)    
        # print("st1.shape",st1.shape)
        # print("st2.shape",st2.shape)            
        
        
        if(st1.shape[0] == 3 and st2.shape[0] == 3 ):
            
            st1 = st1.reshape(-1,100,100,st1.shape[0])
            st2 = st2.reshape(-1,100,100,st2.shape[0])
            vp = np.concatenate((vp, st1), axis=0)
            rof = np.concatenate((rof, st2), axis=0)
            label.append(y)
        else: 
            print("ignored:",k)
            
    vp = vp[1:]
    rof = rof[1:] 

    st = np.concatenate((vp, rof), axis=3)
    # print(st.shape)
    return st,np.array(label)
    
def get_data():
    X_train,y_train,X_val,y_val = top()
    X_train = select(X_train)   
    X_val = select(X_val)     
    # X_val,y_val = MTF(select(X_val),y_val)
    # X_train,y_train = MTF(select(X_train),y_train)    
    # X_val,y_val = MTF(select(X_val),y_val)    
    # print("X_train.shape",X_train.shape) 
    # print("X_val.shape",X_val.shape) 
    # print("y_train",y_train.shape) 
    # print("y_val",y_val.shape)        
    return X_train,y_train,X_val,y_val
    
def main():
    s1 = timeit.default_timer()  
    

    for ii in range(1,14):
        get2sec(ii)
        save_data(ii)
        load_data(ii)
    
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
if __name__ == '__main__':  
    main()
