# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 02:10:04 2021

@author: 006501205
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd 
import time
import timeit
from random import randint
import math
# from unwrap_angle import unwrap_angle

def AppRank(u, s, vh, error):
    
    for r in range(s.shape[0]):
        svd_res = np.linalg.norm(np.dot(np.dot(u,s), vh) - np.dot(np.dot(u[:, :r], s[:r,:r]), vh[:r,:]))
        svd_org = np.linalg.norm(np.dot(np.dot(u,s), vh))

        eta_r = svd_res / svd_org
        if eta_r <= error:
            break

    return r, u[:, :r]

##unwrapping func
def unwrap_angle(df1):
    #input df1 is a dataframe
    df=df1.copy()

    
    df=np.radians(df)
    x=df.sort_index()
    array1=x.iloc[:len(df)-1]
    array2=x.iloc[1:]    
    
    diff=np.array(array2)-np.array(array1)
    idx=np.where(abs(diff)>=np.pi)[0]
    
    for i in idx:
        #print(i)
        x[i+1:]=x[i+1:]+np.pi*2*-1*np.sign(diff[i])
    
    return(x)






def fun(xx):
    xx = xx[:,:,4*3600:7*3600]
    print(xx[0])
    print(xx.shape)
    x1=xx[:,1,:] - xx[:,3,:]
    # plt.plot(x1[0,:])
    # plt.show()

    x = np.zeros((xx.shape[0],xx.shape[1]+2, xx.shape[2]))
    x[:,:-2,:] = xx.copy()

    #Calculated_p = np.zeros([x.shape[0],2,x.shape[2]]) 
    # Calculate Active and Reactive Powers
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            x[i,xx.shape[1],j] = math.sqrt(3) * x[i,0,j] * x[i,2,j] * math.cos(x[i,1,j]-x[i,3,j]) 
            x[i,xx.shape[1]+1,j] = math.sqrt(3) * x[i,0,j] * x[i,2,j] * math.sin(x[i,1,j]-x[i,3,j]) 

    #x2=x[:,6,:] 
    #plt.plot(x2[0,:])
    #plt.show()
    #x3=x[:,7,:] 
    #plt.plot(x3[0,:])
    #plt.show()

    xv_a = x[:,1,:]
    xi_a = x[:,3,:]

    # plt.plot(xv_a[0,:])
    # plt.show()
    # plt.plot(xi_a[0,:])
    # plt.show()

    ## Unwrape voltage and current angles
    for i in range(xv_a.shape[0]):
        df_va = pd.DataFrame(xv_a[i,:].T)
        df_ia = pd.DataFrame(xi_a[i,:].T)
        xva = unwrap_angle(df_va).to_numpy()
        xia = unwrap_angle(df_ia).to_numpy()
        xv_a[i,:] = xva.flatten()
        xi_a[i,:] = xia.flatten()

    # plt.plot(xv_a[0,:])
    # plt.show()
    # plt.plot(xi_a[0,:])
    # plt.show()

    t_window = 30
    miss = 0.2
    # remove nan values
    inds=np.where(np.isnan(x))
    col_mean=[]
    x[inds]=0
    # inserting nan value with interopolation of 5 closest values
    for ii in range(len(inds[0])):
       col_mean.append(np.mean(x[inds[0][ii],inds[1][ii],inds[2][ii]-5:inds[2][ii]+5]))
       x[inds[0][ii],inds[1][ii],inds[2][ii]]=col_mean[ii]
       
    np.random.seed(0)   
                        
    lstnan = np.isnan(x)
    x = np.nan_to_num(x)

    if len(x.shape)==3:           
        X_mat = x.reshape([x.shape[0]*x.shape[1], x.shape[2]]).T

    elif len(x.shape)==2:
        X_mat = x.T

      
    error = 0.01
    errList_OLAP = []
    time_OLAP = []
    Omg_cal = []

    Zeta_m ={}
    for m in range(x.shape[1]):
        eta = []
        zeta = []
        X=x[:,m,:]
        X_mat = X.T
        for i in range(X_mat.shape[0]):
    #        if i % 100 ==0:
    #            print('Iteration '+ str(i)+ ' for measurement ' +str(m))
    #         Intialze the first matrix with no missing entries
            if i == 0:
                Rec_M = X_mat[0:t_window,:]
                Orig_M = Rec_M
                Rec_OLAP = Rec_M
                M_w = X_mat[0:t_window,:]
                for i_k in range(t_window):
                    eta.append(0)
                
            t = i + t_window # proceed with the new sample index
            
            if t == X_mat.shape[0]:
                break
            
            # svd calculations for the old matrix
            u, s, vh = svd(M_w.T)
            eta.append(s[1] / s[0])
        
            
            if eta[t-t_window] != 0:
                zeta_k = (eta[t] - eta[t-t_window]) / (eta[t-t_window] * (t_window))
                zeta.append(zeta_k)
            New_Batch = np.reshape(X_mat[t,:], (-1, 1))
            M_w = np.delete(M_w, (0), axis=0)
            M_w = np.row_stack((M_w, New_Batch.T))
            
        Zeta_m[int(m)] = zeta
    return Zeta_m
#################################################################################    
    
def main():
    s1 = timeit.default_timer()  

    xx= np.load('data/2016_Jan_07_4_Frequency Event_vp_maf.npy')
    # xx= np.load('data/2016_Jan_06_1_XFMR Outage_vp_maf.npy')
    #fig = plt.figure(figsize=(20,16))
    
    Zeta_m = fun(xx)
    fig, axs = plt.subplots(4,2,figsize=(15,10))
    #fig.tight_layout()

    # plt.subplot(1, 2, 1)       
    axs[0,0].plot(Zeta_m[0])
    axs[0,0].set_title('Coefficient eta for voltage magnitude')
    #plt.show()

    # plt.subplot(1, 2, 2) 
    axs[1,0].plot(Zeta_m[4]) 
    axs[1,0].set_title('Coefficient eta for frequency')


    axs[0,1].plot(Zeta_m[2])
    axs[0,1].set_title('Coefficient eta for positive current magnitude')

    # plt.subplot(1, 2, 3)
    axs[1,1].plot(Zeta_m[5])  
    axs[1,1].set_title('Coefficient eta for ROCOF')

    axs[2,0].plot(Zeta_m[6])  
    axs[2,0].set_title('Coefficient eta for Active Power')

    axs[2,1].plot(Zeta_m[7])  
    axs[2,1].set_title('Coefficient eta for Reactive Power')

    # plt.subplot(1, 2, 2) 
    axs[3,0].plot(Zeta_m[1]) 
    axs[3,0].set_title('Coefficient eta for voltage angle')


    x1=xx[:,0,:]
    #plt.plot(x1[0,:])
    axs[3,1].plot(x1[0,:],'r')  
    axs[3,1].set_title('Voltage magnitude signal')
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    plt.tight_layout() 
    plt.show()


if __name__ == '__main__':  
    main()

 
