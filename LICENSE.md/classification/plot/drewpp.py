
# This is the check the modified Z-score
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


def rd2(k,i):
    path1 = '../pickleset/'

    # list = ['ip_m','vp_a','va_m','va_a','vb_m','vb_a','vc_m','vc_a','rocof']
    list = ['rocof','vp_m','ip_m']

    p1 = open(path1 +'X2_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")
    p3  = open(path1 +'y2_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")    
    pk3 = pickle.load(p3)
    # print(pk3.shape)
    # pk3 = deal_label(pk3)
    # print(pk3.shape)
    

    pk3 = pk3[:,1]   
    pk3 = pk3.astype(np.int32)          
    pk1 = pickle.load(p1)


    X_train = pk1
    y_train = pk3
    X_train, y_train  = rm2(X_train, y_train)

    print(X_train.shape) 
    # print(X_val.shape) 
    print(y_train.shape) 
    # print(y_val.shape) 
   

    return X_train, y_train   

def Modified_Z(data):
    c = 1.4826
    median = np.median(data)
    print(median)
    print("median.shape",median.shape)
    dev_med = np.array(data) -median
    print("dev_med.shape",dev_med.shape)
    mad = np.median(np.abs(dev_med))
    z_score = dev_med/(mad*mad)
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
    
    
def do(): 



    i = 1
    fname1 = "gen/set"+str(i)+"x1.npy"
    a = np.load(fname1) 
    
    fname2 = "gen/set"+str(i)+"x2.npy"
    b = np.load(fname2)     
    
    X_train, y_train  = rd2(1,1)
    X_train2, y_train2  = rd2(1,0)

    if((y_train==y_train2).all()):
        # arr = np.zeros([X_train.shape[0],23]) 
        # X_train.shape[0]
        for j in range(0,100):

            y = y_train[j]
            c= int(a[j])
            d= int(b[j])            
        
            plt.figure( figsize=(6,8))
            if(c+120<10800 and c>-1 and d+120<10800 and d>-1):
                for i in range(0,23):
                    
                    x = X_train[j,i,:,:]
                    x2 = X_train2[j,i,:,:]
                    x= np.squeeze(x)
                    x2= np.squeeze(x2)     
                    
                    zz= Modified_Z(x)
                    z2= Modified_Z(x2)       


                    ind = np.argmin(zz)
                    ind2 = np.argmin(z2)
                    
                    k = np.mean(x)                    
                    plt.subplot(4,1, 1)
                    
                    # plt.plot(Hlight,x[Hlight], color=lighten_color('g', 0.2), linewidth=22.0, linestyle="-",zorder=-1)
                    plt.scatter(ind,x[ind],marker='p',c='',edgecolors='r',zorder=10)
                    # plt.scatter(signal.argrelextrema(yvals,np.less)[0],x[signal.argrelextrema(yvals,np.less)[0]],marker='o',c='',edgecolors='r',zorder=10)
                    plt.plot(range(len(x)),x,zorder=0)
                    plt.ylim(0.96*k,1.04*k)  
                    plt.ylabel("Vp_m ")
        

                    
                    plt.subplot(4,1, 2)
                    plt.scatter(ind,zz[ind],marker='p',c='',edgecolors='r',zorder=10)
                    plt.plot(np.arange(0, len(zz)),zz,zorder=0)
                    plt.ylabel("Z score of Vp_m")
                    
                    plt.subplot(4,1, 3)
                    plt.plot(range(len(x2)),x2,zorder=0)
                    plt.scatter(ind2,x2[ind2],marker='p',c='',edgecolors='r',zorder=10)
                    plt.ylabel("ROCOF")
                    plt.title('label: '+str(y))   

                    plt.subplot(4,1, 4)
                    plt.scatter(ind2,z2[ind2],marker='p',c='',edgecolors='r',zorder=10)
                    plt.plot(np.arange(0, len(z2)),z2,zorder=0)
                    # plt.plot(np.arange(0, len(z2)),z2,zorder=0)
                    plt.tight_layout()
                    plt.ylabel("Z score of Rocof")     
                    
                Hlight = range(c,c+120)
                Hlight2 = range(d,d+120)
                plt.subplot(4,1, 1)
                plt.plot(Hlight,x[Hlight], color=lighten_color('g', 0.2), linewidth=22.0, linestyle="-",zorder=-1)
                plt.subplot(4,1, 2)
                plt.plot(Hlight,zz[Hlight], color=lighten_color('g', 0.2), linewidth=22.0, linestyle="-",zorder=-1)
                plt.subplot(4,1, 3)
                plt.plot(Hlight2,x2[Hlight2], color=lighten_color('g', 0.2), linewidth=22.0, linestyle="-",zorder=-1)
                plt.subplot(4,1, 4)
                plt.plot(Hlight2,z2[Hlight2], color=lighten_color('g', 0.2), linewidth=22.0, linestyle="-",zorder=-1)                            
                
                plt.savefig('gen/kk_'+str(j))     
        # 
        print("save done")

def plt_plot():        
    X_train, y_train  = rd2(1,2)
    
    for j in range(0,50):
        plt.figure( figsize=(16,10))
        for i in range(0,1):
            t  =  X_train[j,i,:,:]
    
            plt.subplot(8,3, i+1)
            plt.title("label = "+str(y_train[j]))
            # plt.scatter(ind,t,marker='p',c='',edgecolors='r',zorder=10)
            plt.plot(np.arange(0, t.shape[0]),t,zorder=0)
        for i in range(1,23):
            t  =  X_train[j,i,:,:]
    
            plt.subplot(8,3, i+1)
            # plt.title("label = "+str(y_train[j]))
            # plt.scatter(ind,t,marker='p',c='',edgecolors='r',zorder=10)
            plt.plot(np.arange(0, t.shape[0]),t,zorder=0)        
        plt.tight_layout()    
        plt.savefig('gen/ip_m'+str(j)) 
        
def main():
    s1 = timeit.default_timer()  
    # rd_split()
    do()
    # plt_plot()
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
if __name__ == '__main__':  

    main()
