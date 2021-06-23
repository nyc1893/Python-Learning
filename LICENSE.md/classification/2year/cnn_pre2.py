

# import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import math
import pandas as pd
import time  
from sklearn import metrics  
import pickle as pickle  
import pandas as pd


import pickle
import datapick
import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# python getlabel.py 2>&1 | tee b1.log
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time  
from sklearn import metrics  
import pickle as pickle  


import timeit

from sklearn import metrics  
import pickle as pickle  
import pandas as pd
import sys
import os
from ED_OLAP import get_feature

# from rno_fun import data_lowfreq



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
                # word.append(y[i,0])
                
                
            if y[i,0]==1:
                y_new.append(1)
                X_new.append(X[i,:,:,:])
                # word.append(y[i,0])
        
                
            elif y[i,0]==2:
                y_new.append(2)
                X_new.append(X[i,:,:,:])
                # word.append(y[i,0])
            
            elif y[i,0]==3:
                y_new.append(3)
                X_new.append(X[i,:,:,:])
                # word.append(y[i,0])
            elif y[i,0]==4:
                y_new.append(0)
                X_new.append(X[i,:,:,:])
                # word.append(y[i,0])
            elif y[i,0]==5:
                y_new.append(1)
                X_new.append(X[i,:,:,:])
                # word.append(y[i,0])
            
        
        
    return  np.array(X_new), np.array(y_new)
    
def pack_those1(ii):    

    X_vpm, y_vpm  = rd2(ii,0)
    X, y  = rd2(ii,4)
    p1,p2 = rd_power(ii)
    # y = pd.DataFrame(y)
    # y.columns = ['a']
    
    # dt1 = y[y["a"]==0]
    # dt2 = y[y["a"]==1]
    X = np.concatenate((X, X_vpm), axis=3)
    X = np.concatenate((X, p1), axis=3)
    X = np.concatenate((X, p2), axis=3)

    return X,y

def check_y(ii):    

    X_vpm, y_vpm  = rd2(ii,0)
    X, y  = rd2(ii,4)
    
    if((y_vpm==y).all()):
        print("S"+str(ii)+" yeah")
    
    
def fun1():
    X,y= pack_those1(1)
    for ii in range(2,7+1):
        X2,y2 = pack_those1(ii)

        X = np.concatenate((X, X2), axis=0)
        y = np.concatenate((y, y2), axis=0)
        
    training_set=(X, y)
    pickle_out = open("save1.pickle","wb")
    pickle.dump(training_set, pickle_out, protocol=2)
    pickle_out.close()         
    print("save done S1")    

def fun2():
    X,y= pack_those1(8)
    for ii in range(9,13+1):
        X2,y2 = pack_those1(ii)

        X = np.concatenate((X, X2), axis=0)
        y = np.concatenate((y, y2), axis=0)
        
    training_set=(X, y)
    pickle_out = open("save2.pickle","wb")
    pickle.dump(training_set, pickle_out, protocol=2)
    pickle_out.close()         
    print("save done S2")    
    
    
def see_info(y):    
    y = pd.DataFrame(y)
    y.columns = ['a']
    
    dt1 = y[y["a"]==0]
    dt2 = y[y["a"]==1]
    dt3 = y[y["a"]==2]
    print(dt1.shape)
    print(dt2.shape)    
    print(dt3.shape)
def loading1():        
    p1 = open( "save1.pickle","rb")
    p2 = open( "save2.pickle","rb")
    X, y = pickle.load(p1)
    X2, y2 = pickle.load(p2)
    X = np.concatenate((X, X2), axis=0)
    # for i in range(0,20):
    y = np.concatenate((y, y2), axis=0)
    see_info(y)

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler    

def loading3():    
    p1 = open( "save1.pickle","rb")
    p2 = open( "save2.pickle","rb")
    X, y = pickle.load(p1)
    X2, y2 = pickle.load(p2)
    X = np.concatenate((X, X2), axis=0)
    # for i in range(0,20):
    y_test = np.concatenate((y, y2), axis=0)
    see_info(y)
    ind0 = np.where(y_test==0)[0]
    ind1 = np.where(y_test==1)[0]
    ind2 = np.where(y_test==2)[0]
    
    pca = PCA(n_components=2)
    X_p = pca.fit(X).transform(X)
    ax = plt.figure()
    
    plt.scatter(X_p[ind0,0], X_p[ind0,1], c='r', label="Line")
    plt.scatter(X_p[ind1,0], X_p[ind1,1], c='g', label="Trans")
    plt.scatter(X_p[ind2,0], X_p[ind2,1], c='b', label="Osc")    
    
    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    # plt.title('Iris')
    plt.legend()
    plt.savefig('pca') 
        
    
def loading2():        
    p1 = open( "save1.pickle","rb")
    p2 = open( "save2.pickle","rb")
    X, y = pickle.load(p1)
    X2, y2 = pickle.load(p2)
    X = np.concatenate((X, X2), axis=0)
    y = np.concatenate((y, y2), axis=0)
    path2 = 'index4/'
    tr = np.load(path2+'tr_cc.npy') 
    val = np.load(path2+'val_cc.npy') 
    list1 = tr.astype(int).tolist()
    list2 = val.astype(int).tolist()
    
    X_train = X[list1]
    y_train = y[list1]
    
    c = pd.DataFrame(y_train)
    c.columns = ['a']
    dt2 = c[c["a"]==1].index
    y2 = y_train[dt2]
    X2 = X_train[dt2]
    for i in range(0,12):
        y_train = np.concatenate((y_train, y2), axis=0)
        X_train = np.concatenate((X_train, X2), axis=0)
    # print(dt1.shape)
    # print(dt2.shape)
    
    X_val = X[list2]
    y_val = y[list2]
    
    c = pd.DataFrame(y_val)
    c.columns = ['a']
    dt2 = c[c["a"]==1].index
    y2 = y_val[dt2]    
    X2 = X_val[dt2]  
    for i in range(0,12):
        y_val = np.concatenate((y_val, y2), axis=0)  
        X_val = np.concatenate((X_val, X2), axis=0)
        
    see_info(y_train)
    see_info(y_val)
    
    print("X_train.shape",X_train.shape)
    print("y_train.shape",y_train.shape)
    print("X_val.shape",X_val.shape)
    print("y_val.shape",y_val.shape)
    
    return X_train,y_train,X_val,y_val 
    
def rd_power(ii):
    X_vpm, y_vpm  = rd2(ii,0)
    X_ipm, y_ipm  = rd2(ii,1)
    X_vpa, y_vpa  = rd2(ii,2)
    X_ipa, y_ipa  = rd2(ii,3)
    temp3 =[]
    for i in range(X_vpm.shape[2]):
        temp3.append(4*np.pi*i)

    temp3 = np.array(temp3)
    temp3 = temp3[:,np.newaxis]
    # print("temp3.shape",temp3.shape)
    st1 =[]
    st3 =[]
    # y_vpm.shape[0]
    for j in range(y_vpm.shape[0]):
        st2 = []
        st4 = []
        for i in range(0,23):
            # print("X_vpa[j,i,:,:].shape",X_vpa[j,i,:,:].shape)
            temp1 = math.sqrt(3)*X_vpm[j,i,:,:]*X_ipm[j,i,:,:]*np.cos(np.deg2rad(X_vpa[j,i,:,:]-X_ipa[j,i,:,:]))/1000000
            temp2 = math.sqrt(3)*X_vpm[j,i,:,:]*X_ipm[j,i,:,:]*np.sin(np.deg2rad(X_vpa[j,i,:,:]-X_ipa[j,i,:,:]))/1000000
            
            # print("temp1.shape",temp1.shape)
            st2.append(temp1)
            st4.append(temp2)
        st1.append(st2)
        st3.append(st4)
    st1 = np.array(st1)
    st3 = np.array(st3)
    # print(st1.shape)
    # print(st3.shape)
    return st1,st3
    
def rd2(ii,k):
    path1 = '../pickleset2/'
    list = ['vp_m','ip_m','rocof','f']

    p1 = open(path1 +'X_S'+str(ii)+'_'+str(list[k])+'_6.pickle',"rb")

    pk3  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[k])+'_6.csv')
   
    pk1 = pickle.load(p1)

    X_train = pk1
    # [:,:,80*60:100*60,]
    y_train = pk3.values
    
    # X_train = X_train[:1]
    # y_train = y_train[:1]
    # print(" X_train.shape",X_train.shape)
    # print("Before y_train.shape",y_train.shape)
    X_train, y_train  = rm3(X_train, y_train)
    # print("X_train.shape",X_train.shape)
    # print("After y_train.shape",y_train.shape)
    return X_train, y_train        

def get_split(ii):        
    path = "ml/"
    # df2 = pd.read_csv(path+"savefreq_10m_23_"+str(ii)+".csv")
    df2 = pd.read_csv(path+"savefreq2_23_"+str(ii)+".csv")
    # y = df2["label"].values
    
    # df2 = pd.read_csv(path+"savefreq_10m_23_"+str(ii)+".csv")
    df2['r'] = -df2['upBar']/df2['downBar']
    df2['std'] = df2['upstd']+df2['downstd']
    df2 = df2.replace([np.inf, -np.inf], np.nan)
    df2 = df2.dropna()
    y = df2["label"].values    
    
    spliter(ii,y)

def spliter(num,y3):        
    a = np.arange(0,y3.shape[0])
    tr,val = train_test_split(a,test_size=0.2)   
    print(tr.shape)
    print(val.shape)
    path2 = 'index5/'
    np.save(path2+'tr_'+str(num)+'.npy',tr) 
    np.save(path2+'val_'+str(num)+'.npy',val)

def mid_data(ii):

    path2 = 'index5/'
    tr = np.load(path2+'tr_'+str(ii)+'.npy') 
    val = np.load(path2+'val_'+str(ii)+'.npy') 
    list1 = tr.astype(int).tolist()
    list2 = val.astype(int).tolist()
    path = "ml/"
    # df2 = pd.read_csv(path+"savefreq_10m_23_"+str(ii)+".csv")
    df2 = pd.read_csv(path+"savefreq2_23_"+str(ii)+".csv")
    
    
    df2['r'] = -df2['upBar']/df2['downBar']
    df2['std'] = df2['upstd']+df2['downstd']
    df2.pop('ref')
    df2 = df2.replace([np.inf, -np.inf], np.nan)
    df2 = df2.dropna()
    df2.label[df2.label!=2]=0
    df2.label[df2.label==2]=1
    y = df2.pop("label")
    # print(df2.head())
    X = df2
    
    X_train = X.iloc[list1]
    y_train = y.iloc[list1]
    X_val = X.iloc[list2]
    y_val = y.iloc[list2]

    return X_train,y_train,X_val,y_val
    
def proces(data,k):
    st1 =[]
    data = np.squeeze(data)
    
    for j in range(0,data.shape[0]):
        st2=[]
        for i in range(0,data.shape[1]):
            if(k == 0):
                temp = data[j,i,:]/np.mean(data[j,i,:])
            elif(k == 1):
                temp = np.deg2rad(data[j,i,:])
            elif(k == 2):
                temp = data[j,i,:]/100             
            elif(k == 3):
                temp = np.deg2rad(data[j,i,:])                
            elif(k == 4):
                temp = data[j,i,:]/np.mean(data[j,i,:])            
            elif(k == 5):
                temp = data[j,i,:]      
            st2.append(temp)
        st1.append(st2)

    st1 = np.array(st1)
    st1 = st1[:,:,np.newaxis,:]
    # print(st1.shape)
    return st1
    
def rd3(ii,k):

    path1 = '../pickleset1/'
    list = ['vp_m','vp_a','ip_m','ip_a','f','rocof']
    if(ii!=12):
        p1 = open(path1 +'X_S'+str(ii)+'_'+str(list[k])+'_6.pickle',"rb")
        pk1 = pickle.load(p1)
        
        pk3  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[k])+'_6.csv')
   
    else:
        p1 = open(path1 +'X_S'+str(ii)+'_'+str(list[k])+'_61.pickle',"rb")
        pk1 = pickle.load(p1)
        
        p1 = open(path1 +'X_S'+str(ii)+'_'+str(list[k])+'_62.pickle',"rb")
        pk2 = pickle.load(p1)   
        
        
        pk3  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[k])+'_61.csv')
        pk4  = pd.read_csv(path1 +'y_S'+str(ii)+'_'+str(list[k])+'_62.csv')
        
        pk1= np.concatenate((pk1, pk2), axis=0)
        pk3= pd.concat([pk3, pk4], axis=0)
        
    # X_train = pk1
    # [:,:,0*60:91*60,]
    X_train = pk1[:,:,(300-30)*60:(300+61)*60,]
    
    y_train = pk3.values
    
    
    X_train, y_train  = rm3(X_train, y_train)
    X_train=X_train.transpose(0,1,3,2)
    # print(X_train.shape)
    
    X_train = proces(X_train,k) 
    
    # print(X_train.shape)
    return X_train, y_train 
    
def proces2(data,k):
    st1 =[]
    data = np.squeeze(data)
    
    for j in range(0,data.shape[0]):
        st2=[]
        for i in range(0,data.shape[1]):
            if(k == 0):
                temp = data[j,i,:]/np.mean(data[j,i,:])
            elif(k == 1):
                temp = np.deg2rad(data[j,i,:])
            elif(k == 2):
                temp = data[j,i,:]/np.mean(data[j,i,:])            
            elif(k == 3):
                temp = np.deg2rad(data[j,i,:])                
            elif(k == 4):
                temp = data[j,i,:]/np.mean(data[j,i,:])            
            elif(k == 5):
                temp = data[j,i,:]      
            st2.append(temp)
        st1.append(st2)

    st1 = np.array(st1)
    st1 = st1[:,:,np.newaxis,:]
    # print(st1.shape)
    return st1    


    
def sav(ii):  
    k =0
    a,y_train = rd3(ii,k)
    for k in range(1,5+1):
        a2,_ = rd3(ii,k)
        a = np.concatenate((a, a2), axis=2)  
        
    print(a.shape)
    x = np.zeros((a.shape[0],a.shape[1],a.shape[2]+2, a.shape[3]))
    x[:,:,:-2,:] = a.copy()
    for i in range(x.shape[0]):
        for i2 in range(x.shape[1]):
            for j in range(x.shape[2]):
                x[i,i2,a.shape[2],j] = math.sqrt(3) * x[i,i2,0,j] * x[i,i2,2,j] * math.cos((x[i,i2,1,j])-(x[i,i2,3,j])) 
                x[i,i2,a.shape[2]+1,j] = math.sqrt(3) * x[i,i2,0,j] * x[i,i2,2,j] * math.sin((x[i,i2,1,j])-(x[i,i2,3,j]))        
                    
    print(x.shape)
    
    path3 = 'ml2/event/'
    # save features for all the events
    # for i in range(1,3):
    temp = x[:50]
    pickle_out = open(path3 + "10m_X_S"+str(ii)+"_seg_"+str(1)+".pickle","wb")
    pickle.dump(temp, pickle_out, protocol=2)
    pickle_out.close()    
    
    # temp = x[100:]
    # pickle_out = open(path3 + "10m_X_S"+str(ii)+"_seg_"+str(2)+".pickle","wb")
    # pickle.dump(temp, pickle_out, protocol=2)
    # pickle_out.close()    
        
def pre_olap(ii):  
    k =0
    a,y_train = rd3(ii,k)
    for k in range(1,5+1):
        a2,_ = rd3(ii,k)
        a = np.concatenate((a, a2), axis=2)  
    st = [] 
    
    # a.shape[0]
    


    path3 ='ml2/dif_scal/label/'
    yy = pd.DataFrame(y_train)
    yy.columns = ["label"]
    yy.to_csv(path3 +"y_S"+str(ii)+".csv",index =None)    
    print("save done S"+str(ii))


    
def main():
    s1 = timeit.default_timer()  
    # for i in range(1,14):
        # get_split(i)
        
        
    # num = int(sys.argv[1])
    for num in range(1,13+1):
    # for num in range(7,13+1):
        pre_olap(num)
    # sav(num)
# cd py_file/code/cnn2
    # sav(num)
    # fun2()
    s2 = timeit.default_timer()
    #running time
    print('Time: ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

