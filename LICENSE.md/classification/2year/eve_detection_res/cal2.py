
#This file is located at py_file/code/cnn2/ml2/data_pre/
# import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import math
import pandas as pd
import time  
from sklearn import metrics  
import pickle as pickle  
import timeit
import matplotlib.pyplot as plt
from scipy.linalg import svd 


#get abnormal
def get_ind(temp,scale):
    select =[0,2,4,5,6,7]
    df2 = temp[select,:]
    b = np.zeros(6)
    a =[]
    for j in range(6):
        a.append(np.argmax(df2[j]))
    a.sort()
    
    for i in range(0,6):
        ind = a[i]
        b[i] = num_contain(ind,a,scale)
        
    max = 1
    c = -1
    res =-1
    for i in range(0,6):
        if(b[i]>max):
            max= b[i]
            c=i
            res = a[c]  
           
    if(res==-1):
        ss = np.max(df2,axis=1)
        id = np.argmax(df2[np.argmax(ss)])
        res = id 
    # ck = 5460-scale*2
    if(res>5460-scale*3+1):
        res = 5460-scale*3+1
    # print(res)
    return (res-1)

#get normal
def get_ind2(temp,scale):
    select =[0,2,4,5,6,7]
    df2 = temp[select,:]
    # print(df2.shape)
    b = np.zeros(6)
    b = np.mean(df2,axis = 1)
    res = -1
    i = 0
    for i in range(0,df2.shape[1]-scale):
        temp2 = 5*b - np.max(df2[:,i:i+scale],axis =1) 
        if(( temp2> 0).all()):
            res = i
    arr = 100*np.ones(df2.shape[1]-scale)
    if(res ==-1):
        for i in range(0,df2.shape[1]-scale):
            arr[i] = np.max((np.max(df2[:,i:i+scale])))

        res = np.argmin(arr)
    return (res-1)


    
    
def num_contain(ind,a,scale):
    win=scale
    low = ind-1
    high = low+win
    ll = []
    sum =0
    for i in range(6):
        if(a[i]>low and a[i]<high):
            sum+=1
            
    return sum
    

    

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
    df = pd.read_csv("../../../../../muti.csv")
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

    path1 = '../../../pickleset1/'
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
        
    X_train = pk1
    # [:,:,::30,]
    # [:,:,0*60:91*60,]
    # X_train = pk1[:,:,(300-30)*60:(300+60)*60,]
    
    y_train = pk3.values
    
    
    X_train, y_train  = rm3(X_train, y_train)
    # X_train=X_train.transpose(0,1,3,2)
    # print(X_train.shape)
    
    # X_train = proces(X_train,k) 
    
    # print(X_train.shape)
    return X_train, y_train 





            
def dev_ind(ind2,scale):
    if(ind2-scale<0):
        ind = range(0,2*scale)  
    elif(ind2+scale>5400):
        ind = range(5400 - 2*scale,5400)          
        
    else:
        ind = range(ind2-scale,ind2+scale)  
    return ind
    

def get_single(x,t_window):



    # plt.plot(xv_a[0,:])
    # plt.show()
    # plt.plot(xi_a[0,:])
    # plt.show()

    # t_window = 30
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




      
    error = 0.01
    errList_OLAP = []
    time_OLAP = []
    Omg_cal = []

    Zeta_m =[]
    
    for m in range(0,1):
        eta = []
        zeta = []
        X=x
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
            
        Zeta_m.append(zeta)
    Zeta_m = np.array(Zeta_m)
    print(Zeta_m.shape)
    return Zeta_m            
            
            

    
def save_index(ii):
    scale= 30
    path1 ="../dif_scal/"+str(scale)+"/"
    p1 = open(path1 +'X_S'+str(ii)+'.pickle',"rb")
    pk1 = pickle.load(p1)
    # print(pk1.shape)
    # path2 = "../dif_scal/label/"
    p2 = pd.read_csv(path1 +"y_S"+str(ii)+".csv")
# def cc2():
    # print(p2.shape)
    j = 0
    temp = pk1[j]
    ind2 = get_ind(temp,scale)
    # ind2 = get_ind2(temp,scale)
    # print(ind2)
    st = temp[:,ind2:ind2+scale]   
    st = st[np.newaxis,:,:]
    print(st.shape)
    for j in range(1, pk1.shape[0]):
        temp = pk1[j]
        ind2 = get_ind(temp,scale) 
        if(ind2!=-1):
            temp2 = temp[:,ind2:ind2+scale]
            temp2 = temp2[np.newaxis,:,:] 
            # print(temp2.shape)
            st = np.concatenate((st,temp2))
    # st = np.array(st)
    
    print(st.shape)

    np.save('eve_ind_'+str(ii)+'.npy',st)
    print("S"+str(ii)+" save done")
    
def save_index2(ii):
    scale= 30
    path1 ="../scal/"+str(scale)+"/"
    p1 = open(path1 +'X_S'+str(ii)+'.pickle',"rb")
    pk1 = pickle.load(p1)
    print(pk1.shape)
    # path2 = "../dif_scal/label/"
    p2 = pd.read_csv(path1 +"y_S"+str(ii)+".csv")
    j = 0
    temp = pk1[j]
    ind2 = get_ind(temp,scale)
    # ind2 = get_ind2(temp,scale)
    # print(ind2)
    st = temp[:,ind2:ind2+scale]   
    st = st[np.newaxis,:,:]
    print(st.shape)
    for j in range(1, pk1.shape[0]):
        temp = pk1[j]
        ind2 = get_ind2(temp,scale) 
        if(ind2!=-1):
            temp2 = temp[:,ind2:ind2+scale]
            temp2 = temp2[np.newaxis,:,:] 
            # print(temp2.shape)
            st = np.concatenate((st,temp2))
    # st = np.array(st)
    
    print(st.shape)

    np.save('nor_ind_'+str(ii)+'.npy',st)
    print("S"+str(ii)+" save done")
        
def top():
    for i in range(1,14):
        save_index(i)
        
    for i in range(1,8):
        save_index2(i)
    
def main():
    s1 = timeit.default_timer()  
    # save_index(1)
    top()
    
    s2 = timeit.default_timer()
    print('Time: ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

