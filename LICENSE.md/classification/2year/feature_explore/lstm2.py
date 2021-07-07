import pandas as pd
import numpy as np

from random import random
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, LSTM, Embedding
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.optimizers import Adam
from keras.layers import Activation, Dense, BatchNormalization, TimeDistributed
from keras import regularizers
import pickle as pickle 
import  sys
import warnings

warnings.filterwarnings('ignore')

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
    ll =[0,2,4,5,6,7]
    path1 ="../dif_scal/"+str(scale)+"/"
    p1 = open(path1 +'X_S'+str(ii)+'.pickle',"rb")
    pk1 = pickle.load(p1)
    print(pk1.shape)
    # path2 = "../dif_scal/label/"
    p2 = pd.read_csv(path1 +"y_S"+str(ii)+".csv")
    # print(p2.shape)
    st = []
    for j in range(pk1.shape[0]):
        temp = pk1[j]
        ind2 = get_ind(temp,scale)
        ind = range(ind2,ind2+scale)        
        # print(ind2)
        st.append(temp[:,ind])
    st = np.array(st) 
    st = st[:,ll,:]
    print(st.shape)
    return st,p2.values
    
def rd_non(ii):


    df1 =rd1(ii,10)
    list = [15,20,25,30,35,40,45,50,55,60,90,210,300]
    for i in list:
        df2=rd1(ii,i)
        df1 = pd.concat([df1, df2],axis = 1)
    print(df1.head())
    print(df1.shape)
    path1 ="../dif_scal/"+str(10)+"/"
    p2 = pd.read_csv(path1 +"y_S"+str(ii)+".csv")
    print(p2.shape)    
    df1["label"] = p2
    df1["label"][df1["label"] != 3]=0
    df1["label"][df1["label"] == 3]=1
    df1.to_csv("com_"+str(ii)+".csv",index =None)
    
def rd_eve(ii,scale):
    path1 ="../stat/"
    path2 = 'index2/'
    tr = np.load(path2+'tr_'+str(ii)+'.npy') 
    val = np.load(path2+'val_'+str(ii)+'.npy') 
    list1 = tr.astype(int).tolist()
    list2 = val.astype(int).tolist()
    scale = 30
    X,y = rd1(ii,scale)


    X_train = X[list1]
    y_train = y[list1]
    X_val = X[list2]
    y_val = y[list2]

    return X_train,y_train,X_val,y_val    
    
def rd_eve2(ii):
    path1 ="../stat/"
    path2 = 'index2/'
    
    path3 ="../dif_scal/30/"
    
    tr = np.load(path2+'tr_'+str(ii)+'.npy') 
    val = np.load(path2+'val_'+str(ii)+'.npy') 
    list1 = tr.astype(int).tolist()
    list2 = val.astype(int).tolist()
    df1 = pd.read_csv(path1+"com_"+str(ii)+'.csv')
    dy1 = pd.read_csv(path3 + "y_S"+str(ii)+".csv")
    df1.pop("label")
    y = dy1["label"]
    y = y.values
    y[y!=1] = 0
    
    X = df1.values
    
    X_train = X[list1]
    y_train = y[list1]
    X_val = X[list2]
    y_val = y[list2]

    return X_train,y_train,X_val,y_val  
    
def get_non():
    ii =1

    X_train,y_train,X_val,y_val = rd_non(1)

    for ii in range(2,8):
        X_train2,y_train2,X_val2,y_val2 = rd_non(ii)
        X_train = np.concatenate((X_train, X_train2), axis=0)
        y_train = np.concatenate((y_train, y_train2), axis=0)
        
        X_val = np.concatenate((X_val, X_val2), axis=0)
        y_val = np.concatenate((y_val, y_val2), axis=0)  

    return X_train,y_train,X_val,y_val    
    
def get_eve(scale):
    ii =1

    X_train,y_train,X_val,y_val = rd_eve(1,scale)
    # 14
    for ii in range(2,14):
        X_train2,y_train2,X_val2,y_val2 = rd_eve(ii,scale)
        X_train = np.concatenate((X_train, X_train2), axis=0)
        y_train = np.concatenate((y_train, y_train2), axis=0)
        
        X_val = np.concatenate((X_val, X_val2), axis=0)
        y_val = np.concatenate((y_val, y_val2), axis=0)  
    # X_train = X_train.reshape((X_train.shape[0], -1))
    # X_val = X_val.reshape((X_val.shape[0], -1))
    return X_train,y_train,X_val,y_val
    


def top2():
    X_train,y_train,X_val,y_val = get_eve()
    X_train2,y_train2,X_val2,y_val2 = get_non()
    
    X_train = np.concatenate((X_train, X_train2), axis=0)
    y_train = np.concatenate((y_train, y_train2), axis=0)
    
    X_val = np.concatenate((X_val, X_val2), axis=0)
    y_val = np.concatenate((y_val, y_val2), axis=0)      
    
    print(X_train.shape)
    print(y_train.shape)
    return X_train,y_train,X_val,y_val
    
def data():

    dt1 = get_nor()
    dt2,dt3 = get_eve2()
    x =[]
    real = []
    for i in range(dt1.shape[0]):
        x.append(dt1[i])
        real.append(0)

    
    for i in range(dt2.shape[0]):
        # temp = dt2[i]
        x.append(dt2[i])
        real.append(dt3[i])

    x = np.array(x)
    real = np.array(real)
    return  x,real
    
    
    

    


    
def data_pack(scale):
    
    X_train,y_train,X_val,y_val = get_eve(scale)
    print("X_train: ",X_train.shape)
    a1 = 20
    a2 = 10
    a3 = 5
    model = Sequential()
    model.add(LSTM(a1, input_shape=(X_train.shape[1], X_train.shape[2]),kernel_regularizer=regularizers.l1(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(a2))
    model.add(Activation('relu'))
    model.add(Dense(a3))
    model.add(Activation('tanh'))
    model.add(Dense(1, activation='sigmoid'))
    print ('Compiling...')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


    hist  =  model.fit(X_train, y_train, batch_size=32, nb_epoch=50, validation_split = 0.1, verbose = 1)
    
    score, acc = model.evaluate(X_val,y_val, batch_size=1)
    print("scale= ",scale)
    print('Test score:', score)
    print('Test accuracy:', acc)
    
    y_pred=model.predict_classes(X_val) 
    matrix=confusion_matrix(y_val, y_pred)
    print(matrix)
    class_report=classification_report(y_val, y_pred)
    print(class_report)
    # return acc

import timeit    





def main():


    s1 = timeit.default_timer()  
    data_pack(60)
    
    s2 = timeit.default_timer()  
    print ('Runing time is Hour:',round((s2 -s1)/3600,2))

""" 
"""

if __name__ == "__main__":
    main()


