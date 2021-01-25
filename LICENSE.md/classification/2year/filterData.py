
# This is for Feq CWT^2 use
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import pandas as pd
import pywt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

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

from scipy import signal 
import os
import sys
start = timeit.default_timer()



def removePlanned(X,y):
    """
    THIS FUNCTION REMOVES THE PLANNED EVENTS FROM THE EVENT DATASET
    """
    
    X_new=[]
    y_new=[]
    for i in range(len(y)):
        #print(i)
    
        if y[i]==0:
            y_new.append(0)
            X_new.append(X[i,:,:,:])
    
        elif y[i]==1:
            y_new.append(1)
            X_new.append(X[i,:,:,:])
    
            
        elif y[i]==2:
            y_new.append(2)
            X_new.append(X[i,:,:,:])
        
        elif y[i]==3:
            y_new.append(3)
            X_new.append(X[i,:,:,:])
            
        # elif y[i]==6:
            # y_new.append(6)
            # X_new.append(X[i,:,:,:])
        
        # elif y[i]==7:
            # y_new.append(7)
            # X_new.append(X[i,:,:,:])
        

    return  np.array(X_new), np.array(y_new)

def rm2(X,y):
    """
    THIS FUNCTION REMOVES THE PLANNED EVENTS FROM THE EVENT DATASET
    """
    
    X_new=[]
    y_new=[]
    for i in range(len(y)):
        #print(i)
    
        if y[i]==0:
            y_new.append(0)
            X_new.append(X[i,:,:,:])
    
        elif y[i]==1:
            y_new.append(1)
            X_new.append(X[i,:,:,:])
    
            
        elif y[i]==2:
            y_new.append(2)
            X_new.append(X[i,:,:,:])
        
        elif y[i]==3:
            y_new.append(3)
            X_new.append(X[i,:,:,:])
            
        elif y[i]==4:
            y_new.append(0)
            X_new.append(X[i,:,:,:])
        
        elif y[i]==5:
            y_new.append(1)
            X_new.append(X[i,:,:,:])
        

    return  np.array(X_new), np.array(y_new)
    
def separatePMUs(X,y):
    
    """
    This function separates features and their corresponding labels for each PMU
    to make more events 
    """
    
    num_case=X.shape[0]
    num_pmu=X.shape[1]
    num_sample=X.shape[2]
    X_new=[]
    X=X.reshape(num_case*num_pmu,num_sample)
    y2=[]
    for i in range(len(y)):
        if y[i]==0:
            for j in range(num_pmu):
                y2.append(0)

        elif y[i]==1:
            for j in range(num_pmu):
                y2.append(1)
                
        elif y[i]==2:
            for j in range(num_pmu):
                y2.append(2)
                
        elif y[i]==3:
            for j in range(num_pmu):
                y2.append(3)
        elif y[i]==6:
            for j in range(num_pmu):
                y2.append(6)       
        elif y[i]==7:
            for j in range(num_pmu):
                y2.append(7)    
    return X,np.array(y2)

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
    
def deal_label(y_test):
    path1 = 'pca_plot2/'
    y_test = pd.DataFrame(y_test)
    # print(y_test.head())
    y_test.columns = ['event',	'label']
    df = y_test.pop('event')
    # print(df.head())
    df = df.str.split('_')
    df2 = pd.DataFrame(df)
    y_test['year'] = df2['event'].str[0]
    y_test['month'] = df2['event'].str[1]
    y_test['day'] = df2['event'].str[2].astype(int)
    y_test['no'] = df2['event'].str[3]   

    y_test['new'] = y_test['year'].astype(str).str.cat(y_test['month'].astype(str),sep = '_')
    y_test['new'] = y_test['new'].str.cat(y_test['day'].astype(str),sep = '_')
    y_test['new'] = y_test['new'].str.cat(y_test['no'].astype(str),sep = '_')    

    y_test = y_test[['new','label']]
    
    

    df2 = pd.read_csv(path1+'trans2.csv')
    list = df2['0'].tolist()
    y_test['label'] = y_test['label'].astype("int")
    ind1 = y_test['new'].isin(list).tolist()
    # print(ind1)
    df2 = pd.read_csv(path1+'feq2.csv')
    list = df2['0'].tolist()
    ind2 = y_test['new'].isin(list).tolist()
    
    y_test.loc[ind1, 'label'] = 6
    y_test.loc[ind2, 'label'] = 7
    # y_test['label'].iloc[ind2] = 7
    
    
    y_test.pop('new')
    
    # print(y_test.loc[40:55])
    # y_test.to_csv('kankan.csv',index =None)
    return y_test.values

def rd2(k,i):
    path1 = '../pickleset/'

    # list = ['ip_m','vp_a','va_m','va_a','vb_m','vb_a','vc_m','vc_a','rocof']
    list = ['rocof','vp_m']

    p1 = open(path1 +'X2_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")
    p3  = open(path1 +'y2_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")    
    pk3 = pickle.load(p3)
    # print(pk3.shape)
    # pk3 = deal_label(pk3)
    # print(pk3.shape)
    
    
    pk3 = pk3[:,1]   
    pk3 = pk3.astype(np.int32)      
    # print(pk3[:5])
    pk1 = pickle.load(p1)

    # fps=60

    # start_crop=int(fps*60*4)
    # stop_crop=int(fps*60*7)

    # pk1=pk1[:,:,start_crop:stop_crop,:]
    X_train = pk1
    y_train = pk3
    X_train, y_train  = rm2(X_train, y_train)

    # print(X_train.shape) 
    # print(X_val.shape) 
    # print(y_train.shape) 
    # print(y_val.shape) 
   

    return X_train, y_train    
    
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
            for i in range(0,23):
                
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
            for ii in range (0, 89):    
                if(p1<point[ii]):
                    p1 = point[ii]
                    ind = ii
            ar_x1[j] = ind*120
            
            p2=0    
            ind2 =0
            for ii in range (0, 89):    
                if(p2<point2[ii]):
                    p2 = point2[ii]
                    ind2 = ii
            ar_x2[j] = ind2*120
            
            
            
    np.save(fname+"x1.npy",ar_x1)
    np.save(fname+"x2.npy",ar_x2)
    print("save done")


    

            
def diff(listA,listB):
    #求交集的两种方式
    retA = [i for i in listA if i in listB]
    # retB = list(set(listA).intersection(set(listB)))            
    if retA:
        # print(retA)
        return True 
    else:
        return False    

def get_2sec_start():
    for i in range(1,13+1):
        X_train,  y_train   = rd2(i,1)
        X_train2,  y_train2  = rd2(i,0)
        fname = "gen/set"+str(i)
        # cal3(X_train, y_train,X_train2, y_train2,fname) 
        cal4(X_train, y_train,X_train2, y_train2,fname) 
        
def get_split(num,y3):        
    a = np.arange(0,y3.shape[0])
    tr,val = train_test_split(a,test_size=0.2)   
    print(tr.shape)
    print(val.shape)
    path2 = 'index4/'
    np.save(path2+'tr_'+str(num)+'.npy',tr) 
    np.save(path2+'val_'+str(num)+'.npy',val)
            
def fil_Data(X_train, y_train,X_train2, y_train2,i): 
    fname1 = "gen/set"+str(i)+"x1.npy"
    a = np.load(fname1) 
    
    fname2 = "gen/set"+str(i)+"x2.npy"
    b = np.load(fname2)     
    

    path2 = 'index4/'
    tr = np.load(path2+'tr_'+str(i)+'.npy') 
    val = np.load(path2+'val_'+str(i)+'.npy') 
    # print("tr.shape",tr.shape)
    list1 = tr.astype(int).tolist()
    list2 = val.astype(int).tolist()
    sum1 =0
    sum2 =0
    if((y_train==y_train2).all()):
        # arr = np.zeros([X_train.shape[0],23]) 
        vp = X_train[0,0,1:1+120,:].reshape(-1,120)
        vp2 = X_train[0,0,1:1+120,:].reshape(-1,120)
        rof = X_train[0,0,1:1+120,:].reshape(-1,120)
        rof2 = X_train[0,0,1:1+120,:].reshape(-1,120)
        label = []
        label2 = []
        for j in range(0,X_train.shape[0]):

            for i in range(0,23):

                c= int(a[j])
                d= int(b[j])
                if(c+120<10800 and c>-1 and d+120<10800 and d>-1):

                    y = y_train[j]
                    x = X_train[j,i,c:c+120,:].reshape(-1,120)
                    x2 = X_train2[j,i,d:d+120,:].reshape(-1,120)
                    # print(y)

                    if(j in list1 and x is not None):        

                        vp = np.concatenate((vp, x), axis=0)
                        rof = np.concatenate((rof, x2), axis=0)
                        label.append(y)
                        sum1+=1
                    elif(j in list2 and x is not None):
                        vp2 = np.concatenate((vp2, x), axis=0)
                        rof2 = np.concatenate((rof2, x2), axis=0)      
                        label2.append(y)
                        sum2+=1
      
        label = np.array(label)   
        label2 = np.array(label2)        

        vp = vp[1:]
        vp2= vp2[1:]
        rof = rof[1:]
        rof2= rof2[1:]    
        
    # print("fil_Data:")    
    # print("vp",vp.shape)
    # print("rof",rof.shape)
    # print("label",label.shape)
    # print("vp2",vp2.shape)
    # print("rof2",rof2.shape) 
    # print("label2",label2.shape)  
    
        
    return vp,rof,label,vp2,rof2,label2
    
def fil_Data2(X_train, y_train,X_train2, y_train2,i): 
    fname = "gen/set"+str(i)+".npy"
    a = np.load(fname) 

    path2 = 'index4/'
    tr = np.load(path2+'tr_'+str(i)+'.npy') 
    val = np.load(path2+'val_'+str(i)+'.npy') 
    print("y_train.shape",y_train.shape)
    list1 = tr.astype(int).tolist()
    list2 = val.astype(int).tolist()
    sum1 =0
    sum2 =0
    
    if((y_train==y_train2).all()):
        # arr = np.zeros([X_train.shape[0],23]) 
        vp = X_train[0,:,1:1+120,:].reshape(-1,23,120)
        vp2 = X_train[0,:,1:1+120,:].reshape(-1,23,120)
        rof = X_train[0,:,1:1+120,:].reshape(-1,23,120)
        rof2 = X_train[0,:,1:1+120,:].reshape(-1,23,120)
        label = []
        label2 = []
        for j in range(0,X_train.shape[0]):
            c= int(a[j])
            if(c+120<10800 and c>-1):

                y = y_train[j]
                x = X_train[j,:,c:c+120,:].reshape(-1,23,120)
                x2 = X_train2[j,:,c:c+120,:].reshape(-1,23,120)
                #print(y)

                if(j in list1 and x is not None):        

                    vp = np.concatenate((vp, x), axis=0)
                    rof = np.concatenate((rof, x2), axis=0)
                    label.append(y)
                    sum1+=1
                elif(j in list2 and x is not None):
                    vp2 = np.concatenate((vp2, x), axis=0)
                    rof2 = np.concatenate((rof2, x2), axis=0)      
                    label2.append(y)
                    sum2+=1
      
        label = np.array(label)   
        label2 = np.array(label2)        

        vp = vp[1:]
        vp2= vp2[1:]
        rof = rof[1:]
        rof2= rof2[1:]    
        

        
        # print(X_train.shape[0])
        
        # print(len(list1))
        # print(len(list2))
        
        print(sum1)
        print(sum2)        
        
    # print("fil_Data2:")    
    # print("vp",vp.shape)
    # print("rof",rof.shape)
    # print("label",label.shape)
    # print("vp2",vp2.shape)
    # print("rof2",rof2.shape) 
    # print("label2",label2.shape)        
    return vp,rof,label,vp2,rof2,label2
        
def dataPack():

    i=10
    X_train,  y_train   = rd2(i,1)
    X_train2,  y_train2  = rd2(i,0)

    vp,rof,label,vp2,rof2,label2= fil_Data(X_train, y_train,X_train2, y_train2,i)
    # print("vp.shape",vp.shape)
    for i in range(11,13+1):
        X_train,  y_train   = rd2(i,1)
        X_train2,  y_train2  = rd2(i,0)

        vpp,roff,labell,vpp2,roff2,labell2= fil_Data(X_train, y_train,X_train2, y_train2,i)
        vp = np.concatenate((vp, vpp), axis=0)
        rof = np.concatenate((rof, roff), axis=0)
        label = np.concatenate((label, labell), axis=0)
        vp2 = np.concatenate((vp2, vpp2), axis=0)
        rof2 = np.concatenate((rof2, roff2), axis=0)
        label2 = np.concatenate((label2, labell2), axis=0)
        
    print("vp",vp.shape)
    print("rof",rof.shape)
    print("label",label.shape)
    print("vp2",vp2.shape)
    print("rof2",rof2.shape) 
    print("label2",label2.shape)
    return vp,rof,label,vp2,rof2,label2
    
def clean(vp,rof,label):
    x1 = []
    x2 = []
    y = []
    for i in range(0,vp.shape[0]):
        temp1 = vp[i]
        temp2 = rof[i]
        if(temp1[0]!=temp1[-1] and temp2[0]!=temp2[-1]):
            x1.append(temp1)
            x2.append(temp2)
            y.append(label[i])
        else:
            if(np.max(temp1)!=np.min(temp1) and np.max(temp2)!=np.min(temp2)):
                x1.append(temp1)
                x2.append(temp2)
                y.append(label[i])  
    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)
    return x1,x2,y
    
        
def deal_mtf(data,size):
    fold =  data.shape[0]//1000
    res =  data.shape[0]%1000
    mtf = MarkovTransitionField(image_size=size)
    
    temp = mtf.fit_transform(data[:1000])
    for i in range(1, fold):
        temp2 = mtf.fit_transform(data[1000*i:1000*(i+1)])
        temp = np.concatenate((temp, temp2), axis=0)
        
    temp2 = mtf.fit_transform(data[1000*fold:1000*fold+res])
    temp = np.concatenate((temp, temp2), axis=0)

    return temp  

def check(data):
    a = set()
    for ii in np.squeeze(data):
        a.add(ii)
    if( len(a)== 1):
        return 0
    return 1
    
def deal_mtf2(data,data2,label,size):   

    mtf = MarkovTransitionField(image_size=size)
    
    list = []

    j =0
    dd = data[j,0,:].reshape(-1,120)
    dd2 = data2[j,0,:].reshape(-1,120)
    temp3 = mtf.fit_transform(dd).reshape(1,size,size,-1)
    tempp3 = mtf.fit_transform(dd2).reshape(1,size,size,-1)
    for i in range(1,23):
        
        dd = data[j,i,:].reshape(-1,120)
        dd2 = data[j,i,:].reshape(-1,120)
        temp2 = mtf.fit_transform(dd).reshape(1,size,size,-1)
        tempp2 = mtf.fit_transform(dd).reshape(1,size,size,-1)
        temp3 = np.concatenate((temp3, temp2), axis=3)
        tempp3 = np.concatenate((tempp3, tempp2), axis=3) 
    print("temp3",temp3.shape)

    for j in range(1,data.shape[0]):
        
        dd = data[j,0,:].reshape(-1,120)
        dd2 = data2[j,0,:].reshape(-1,120)
        if(check(dd) == 0 or check(dd2)==0):
            list.append(j)
            continue
        temp = mtf.fit_transform(dd).reshape(1,size,size,-1)
        tempp = mtf.fit_transform(dd2).reshape(1,size,size,-1)
        for i in range(1,23):
            # if(np.size(data[j,i,:]) ==np.size(data[0,0,:]) ):
            
            dd = data[j,i,:].reshape(-1,120)
            dd2 = data2[j,0,:].reshape(-1,120)
            if(check(dd) == 0 or check(dd2)==0):
                list.append(j)
                break
            temp2 = mtf.fit_transform(dd).reshape(1,size,size,-1)
            tempp2 = mtf.fit_transform(dd2).reshape(1,size,size,-1)
            # print("shape:",temp2.shape)
            temp = np.concatenate((temp, temp2), axis=3) 
            tempp = np.concatenate((tempp, tempp2), axis=3)
        if(temp.shape[3]==23 and tempp.shape[3] ==23):
            # print(i)
            # print(j)
            # print("temp",temp.shape)
            # print("temp3",temp3.shape)
            temp3 = np.concatenate((temp3, temp), axis=0)
            tempp3 = np.concatenate((tempp3, tempp), axis=0)
    
    res=[]
    for i in range(label.shape[0]):
        if(i in list):
            continue
        res.append(label[i])
    res = np.array(res)
    print(list)

    return temp3,tempp3,res     

def MTF2():

    vp,rof,label,vp2,rof2,label2 = doit()
    size = 120

    vp,rof,label = deal_mtf2(vp,rof,label,size)
    vp2,rof2,label2 = deal_mtf2(vp2,rof2,label2,size)

    print("After MTF: ",vp.shape)    
    print("vp",vp.shape)
    print("rof",rof.shape)
    print("label",label.shape)
    print("vp2",vp2.shape)
    print("rof2",rof2.shape) 
    print("label2",label2.shape)
    
    data = (vp,rof,label,vp2,rof2,label2)
    file = open('gen/pack1', 'wb')
    pickle.dump(data, file)
    file.close()
    print("pickle save done!")    
    
def MTF():

    vp,rof,label,vp2,rof2,label2 = dataPack()
    size = 120
    vp,rof,label = clean(vp,rof,label)
    vp2,rof2,label2 = clean(vp2,rof2,label2)
    
    vp = deal_mtf(vp,size)
    vp2 = deal_mtf(vp2,size)
    rof = deal_mtf(rof,size)
    rof2 = deal_mtf(rof2,size)
    
    print("After MTF: ",vp.shape)    
    print("vp",vp.shape)
    print("rof",rof.shape)
    print("label",label.shape)
    print("vp2",vp2.shape)
    print("rof2",rof2.shape) 
    print("label2",label2.shape)
    data = (vp,rof,label,vp2,rof2,label2)
    file = open('gen/pack_3', 'wb')
    pickle.dump(data, file)
    file.close()
    print("pickle save done!")    
    
def doit():
    i=1
    X_train,  y_train   = rd2(i,1)
    X_train2,  y_train2  = rd2(i,0)

    vp,rof,label,vp2,rof2,label2= fil_Data2(X_train, y_train,X_train2, y_train2,i)
    for i in range(2,6+1):
        X_train,  y_train   = rd2(i,1)
        X_train2,  y_train2  = rd2(i,0)

        vpp,roff,labell,vpp2,roff2,labell2= fil_Data2(X_train, y_train,X_train2, y_train2,i)
        vp = np.concatenate((vp, vpp), axis=0)
        rof = np.concatenate((rof, roff), axis=0)
        label = np.concatenate((label, labell), axis=0)
        vp2 = np.concatenate((vp2, vpp2), axis=0)
        rof2 = np.concatenate((rof2, roff2), axis=0)
        label2 = np.concatenate((label2, labell2), axis=0)    
    
    print("doit")
    print("vp",vp.shape)
    print("rof",rof.shape)
    print("label",label.shape)
    print("vp2",vp2.shape)
    print("rof2",rof2.shape) 
    print("label2",label2.shape)
    return vp,rof,label,vp2,rof2,label2
    
def read_ind():
    i = 1
    fname = "gen/set"+str(i)+"2.npy"
    a = np.load(fname) 
    print(a[:20])
    
def main():
    s1 = timeit.default_timer()  

    # for i in range(1,14):
        # X_train,  y_train   = rd2(i,1)
        # get_split(i,y_train)  
    
    # doit()
    # dataPack()
    MTF()
    # 
    # get_2sec_start()
    # MTF2()

    # rd2(1,1)
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
if __name__ == '__main__':  

    main()
