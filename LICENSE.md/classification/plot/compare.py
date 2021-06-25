

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
    
def rd1(ii,scale):
    # list = [10,15]
    # scale= 10
    path1 ="../dif_scal/"+str(scale)+"/"
    p1 = open(path1 +'X_S'+str(ii)+'.pickle',"rb")
    pk1 = pickle.load(p1)
    print(pk1.shape)
    # path2 = "../dif_scal/label/"
    p2 = pd.read_csv(path1 +"y_S"+str(ii)+".csv")
    print(p2.shape)
    st = []
    for j in range(pk1.shape[0]):
        temp = pk1[j]
        ind2 = get_ind(temp,scale)
        ind = range(ind2,ind2+scale)        
        # print(ind2)
        st.append(temp[:,ind])
    st = np.array(st)    
    st2 =[]
    for j in range(st.shape[0]):
        kk = np.zeros(24)
        for i in range(0,8):
            temp = st[j,i,:]
            kk[i*3+0] =np.max(temp)
            kk[i*3+1] =np.mean(temp)
            kk[i*3+2] =np.std(temp)
        st2.append(kk)

    
    st2= np.array(st2)
    df =pd.DataFrame(st2)
    list1 = ["_"+str(scale)+"_max","_"+str(scale)+"_avg","_"+str(scale)+"_std"]
    list2 = [*range(1,9)]
    list3 = []

    
    for j in range(len(list2)):
        for i in range(len(list1)):
            list3.append(str(list2[j])+list1[i])
    df.columns = list3
    # df["label"] = y.values
    # print(df.head())
    # print(df.shape)
    return df


    

    
def fun(ii):
    df1 =rd1(ii,10)
    list = [15,20,25,30,35,40,45,50,55,60,90]
    for i in list:
        df2=rd1(ii,i)
        df1 = pd.concat([df1, df2],axis = 1)
    print(df1.head())
    print(df1.shape)
    path1 ="../dif_scal/"+str(10)+"/"
    p2 = pd.read_csv(path1 +"y_S"+str(ii)+".csv")
    print(p2.shape)    
    df1["label"] = p2
    df1.to_csv("com_"+str(ii)+".csv",index =None)
    
    
    
def rd2(ii):
    df1 = pd.read_csv("com_"+str(ii)+".csv")
    df1["label"][df1["label"]!=3]=0
    df1["label"][df1["label"]==3]=1
    return df1
    
def top():
    df1 = rd2(1)
    for i in range(2,14):
        df2=rd2(i)
        df1 = pd.concat([df1, df2],axis = 0)
    # print(df1.head())
    # print(df1.shape)
    return df1
 
 
def mid_data(ii):

    path2 = 'index4/'
    tr = np.load(path2+'tr_'+str(ii)+'.npy') 
    val = np.load(path2+'val_'+str(ii)+'.npy') 
    list1 = tr.astype(int).tolist()
    list2 = val.astype(int).tolist()
    path = "ml/"
    
    X = pd.read_csv("com_"+str(ii)+".csv")
    X["label"][X["label"]!=3]=0
    X["label"][X["label"]==3]=1
    
    
    y =X.pop("label")
    pcaa = 1
    if pcaa ==1:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=10)
        X = pca.fit_transform(X)
        X_train = X[list1]
        y_train = y.iloc[list1]
        X_val = X[list2]
        y_val = y.iloc[list2]      
        
        X_train = pd.DataFrame(X_train)
        # y_train = pd.DataFrame(y_train)
        X_val = pd.DataFrame(X_val)
        # y_val = pd.DataFrame(y_val)
        
    else:
        X_train = X.iloc[list1]
        y_train = y.iloc[list1]
        X_val = X.iloc[list2]
        y_val = y.iloc[list2]

    return X_train,y_train,X_val,y_val    
    
def top2():
    X_train,y_train,X_val,y_val = mid_data(1)
    for ii in range(2,14):
        X_train2,y_train2,X_val2,y_val2 = mid_data(ii)
        X_train = pd.concat([X_train, X_train2], axis=0)
        y_train = pd.concat([y_train, y_train2], axis=0)
        
        X_val = pd.concat([X_val, X_val2], axis=0)
        y_val = pd.concat([y_val, y_val2], axis=0)   
    
        
    return X_train,y_train,X_val,y_val          
    
def plot(j):
    plt.figure( figsize=(5,4))
    from sklearn.decomposition import PCA
    X = top()
    
    y_test = X.pop("label").values
    
    list1 = [10,15,20,25,30,35,40,45,50,55,60,90]
    list2 =[]
    for i in range(len(list1)):
        list2.append(str(j)+"_"+str(list1[i])+"_max")
        
    print(list2)
    X = X.loc[:, list2]
    
    
    ind0 = np.where(y_test==0)[0]
    ind1 = np.where(y_test==1)[0]
    pca = PCA(n_components=2)
    
    
    newData = pca.fit_transform(X)
    # print(newData.shape)
    plt.scatter(newData[ind0,0],newData[ind0,1],label='Non-osc')
    plt.scatter(newData[ind1,0],newData[ind1,1],label='Osc')
    plt.legend(loc='best')
    plt.tight_layout()             
    plt.savefig("com_"+str(j)) 
    
def main():
    s1 = timeit.default_timer() 
    for j in range(1,9):
        plot(j)
    s2 = timeit.default_timer()
    print('Time: ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

