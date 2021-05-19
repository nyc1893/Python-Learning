
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
import statsmodels.api as sm 
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


def rm3(X,y):
    """
    THIS FUNCTION REMOVES THE PLANNED EVENTS FROM THE EVENT DATASET
    """
    
    X_new=[]
    y_new=[]
    word = []
    # print(y[1,1])
    for i in range(len(y)):
        #print(i)
        
        if y[i,1]=='0':
            y_new.append(0)
            X_new.append(X[i,:,:,:])
            word.append(y[i,0])
            
            
        elif y[i,1]=='1':
            y_new.append(1)
            X_new.append(X[i,:,:,:])
            word.append(y[i,0])
    
            
        elif y[i,1]=='2':
            y_new.append(2)
            X_new.append(X[i,:,:,:])
            word.append(y[i,0])
        
        elif y[i,1]=='3':
            y_new.append(3)
            X_new.append(X[i,:,:,:])
            word.append(y[i,0])
        elif y[i,1]=='4':
            y_new.append(0)
            X_new.append(X[i,:,:,:])
            word.append(y[i,0])
        elif y[i,1]=='5':
            y_new.append(1)
            X_new.append(X[i,:,:,:])
            word.append(y[i,0])

    return  np.array(X_new), np.array(y_new),np.array(word)




#contains words:
def rd3(k,i):
    path1 = '../pickleset/'

    # list = ['ip_m','vp_a','va_m','va_a','vb_m','vb_a','vc_m','vc_a','rocof']
    list = ['rocof','vp_m','ip_m']

    p1 = open(path1 +'X2_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")
    p3  = open(path1 +'y2_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")    
    pk3 = pickle.load(p3)
    # print(pk3.shape)
    # pk3 = deal_label(pk3)
    # print(pk3.shape)
    

    # pk3 = pk3[:,1]   
    # pk3 = pk3[:,1].astype(np.int32)          
    pk1 = pickle.load(p1)


    X_train = pk1
    y_train = pk3
    X_train, y_train,word  = rm3(X_train, y_train)

    print(X_train.shape) 
    # print(X_val.shape) 
    print(y_train.shape) 
    print(word.shape) 
    print(word[:1])

    return X_train, y_train,word  


list = ['rocof','vp_m','ip_m','vp_a','ip_a']
    
    
def rd2(k,i):
    path1 = '../pickleset2/'

    # list = ['ip_m','vp_a','va_m','va_a','vb_m','vb_a','vc_m','vc_a','rocof']
    

    p1 = open(path1 +'X_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")
    pk3  = pd.read_csv(path1 +'y_S'+str(k)+'_'+str(list[2])+'_6.csv')
 
    pk1 = pickle.load(p1)
    pk3 = np.array(pk3)
    pk3 = pk3[:,1]
    X_train = pk1
    y_train = pk3
    # X_train, y_train  = rm2(X_train, y_train)

    print(X_train.shape) 
    # print(X_val.shape) 
    print(y_train.shape) 
    return X_train, y_train   

def rd3(k,i):
    path1 = '../pickleset2/'

    # list = ['ip_m','vp_a','va_m','va_a','vb_m','vb_a','vc_m','vc_a','rocof']
    # list = ['rocof','vp_m','ip_m']

    p1 = open(path1 +'X_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")
    pk3  = pd.read_csv(path1 +'y_S'+str(k)+'_'+str(list[2])+'_6.csv')
 
    pk1 = pickle.load(p1)
    pk3 = np.array(pk3)
    pk3 = pk3[:,2]
    X_train = pk1
    y_train = pk3
    # X_train, y_train  = rm2(X_train, y_train)

    print(X_train.shape) 
    # print(X_val.shape) 
    print(y_train.shape) 
    return X_train, y_train   
    
def rd4(k,i):
    path1 = '../pickleset2/'

    # list = ['ip_m','vp_a','va_m','va_a','vb_m','vb_a','vc_m','vc_a','rocof']
    # list = ['rocof','vp_m','ip_m']

    p1 = open(path1 +'X_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")
    pk3  = pd.read_csv(path1 +'y_S'+str(k)+'_'+str(list[2])+'_6.csv')
 
    pk1 = pickle.load(p1)
    pk3 = np.array(pk3)
    pk3 = pk3[:,0]
    X_train = pk1
    y_train = pk3
    # X_train, y_train  = rm2(X_train, y_train)

    print(X_train.shape) 
    # print(X_val.shape) 
    print(y_train.shape) 
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
    


def rd_split(): 

    i = 1
    fname1 = "gen/set"+str(i)+"x1.npy"
    a = np.load(fname1) 
    
    fname2 = "gen/set"+str(i)+"x2.npy"
    b = np.load(fname2)    
    a = a[:,np.newaxis]
    b = b[:,np.newaxis]
    a  = np.concatenate((a, b), axis=1)
    np.savetxt("testsplit.txt",a) 
    print("save done")
    
def decode(a):

    list = ["Line+Transformer","Line+Frequency","Line+Osillation","Line+Frequency+Transformer"]


    if(a == 4):
        return list[0]
    elif(a==5):
        return list[1]
        
    elif(a==6):
        return list[2]
    elif(a==7):
        return list[3]
        
def static(a):
    list = []
    for i in range(0,23):
        b= a[i,:,:]
        list.append(np.max(b)-np.min(b))
    return str(np.max(np.array(list))),str(np.mean(np.array(list)))    
# 23 PMU separate:    
def plt_3(ii,v): 

    # ii = 3
    fname1 = "gen/set"+str(ii)+"x1.npy"
    a = np.load(fname1) 
    
    fname2 = "gen/set"+str(ii)+"x2.npy"
    b = np.load(fname2)     
    
    X_train, y_train  = rd2(ii,1)
    X_train2,ll   = rd4(ii,0)
    X_train3,w  = rd3(ii,2)
    # print("y_train.shape",y_train.shape)
    cnt = 0
    # df = pd.read_csv("../../../muti.csv")
    # X_train.shape[0]
    # print(type(ll[0]))
    for j in range(0, X_train.shape[0]):

        y = y_train[j]
        lei = ll[j]
        # print(lei)
        c= int(a[j])
        d= int(b[j])            
        word = w[j]
        plt.figure(figsize=(15,8))
        # print(X_train[j,:,:,:].shape)
        # vp = np.mean(X_train[j,:,:,:])
        # print("mean: ",vp)
        aa,bb= static(X_train[j,:,:,:])
        cc = "Max "+aa+", mean "+bb
        
        num = 10700
              
        
        if(c+120<10800 and c>-1 and d+120<10800 and d>-1 and(v in y )and cnt<5):
        # if(c+120<10800 and c>-1 and d+120<10800 and d>-1 and(lei==v )  and cnt<50):
            vp = np.mean(X_train[j,:,:,:])
            b = np.sum(X_train[j,:,:,:],axis = 0)                
            b = np.squeeze(b)            
            bb = np.zeros(10800)
            for i in range(0,23):
                
                x = X_train[j,i,:,:]/vp
            
                x2 = X_train2[j,i,:,:]
                x3 = X_train3[j,i,:,:]/np.mean(X_train3[j,i,:,:])

                k = np.mean(x)    
                if(np.mean(x)>0.8):
                    plt.subplot(3,1,1)
                    
                    plt.plot(range(len(x)),x,linewidth=1)
                    # plt.ylim(0.97*k,1.005*k)  
                    plt.ylabel("Vp_m ")
        
                    # if(lei>3):
                        # plt.title(decode(lei)+" "+word+" "+str(y)+"_"+ cc)
                    # else:
                        # plt.title(word+" "+str(y)+"_"+ cc)   
                  
                    plt.subplot(3,1,2)
                    plt.plot(range(len(x2)),x2,linewidth=1)
                    # plt.scatter(ind2,x2[ind2],marker='p',c='',edgecolors='r',zorder=10)
                    plt.ylabel("ROCOF")

                    plt.subplot(3,2,5)
                    plt.plot(range(len(x3)),x3,linewidth=1)
                    # plt.scatter(ind2,x2[ind2],marker='p',c='',edgecolors='r',zorder=10)
                    plt.ylabel("Ip_m")
                    

            # plt.text(10,5,word,zorder=22)
            # print(word)
            L = 1
            for i in range(0,num):
                bb[i] = b[i+L]-b[i]
            sample = bb
            ecdf = sm.distributions.ECDF(sample)
            x = np.linspace(0, max(sample))
            y = ecdf(x)    
            plt.subplot(3,2,6)
            plt.plot(x,y,label='L = '+str(L))    
            plt.xlabel('23 PMUs vp_m',fontsize=11)
            plt.ylabel('CDF',fontsize=9)  

         
            plt.tight_layout() 
            plt.savefig('gen/tri_label='+str(v)+'_'+str(cnt))   
            # plt.savefig('gen/tri_'+v+'_'+str(cnt))     
            cnt = cnt+1
            plt.close('all')
    print("save done")
    
    # list = ['rocof','vp_m','ip_m','vp_a','ip_a']

def test():
    text2 = "2016_Apr_22_2"
    df = pd.read_csv("../../../muti.csv")
    # print(df.head())
    
    print(df['v'][df['new'] == text2].values == 7)
def main():
    s1 = timeit.default_timer()  
    # rd_split()
    # do()
    # plt_plot()
    
    ii =3
    # v = "Frequency"
    # v = "Line_Weather"
    # v = "Line_Lightning"
    # v = "Line_Equipment"
    v = "Line"
    # v = "Oscillation"
    # rd2(1,2)
    # plt_vp_m(ii,v)
    # plt_rocof(ii,v)
    # v = 2
    # cnt = 0
    plt_3(ii,v)


    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
if __name__ == '__main__':  

    main()
