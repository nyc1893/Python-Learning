
# testing mutilabeled DNN model

import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import math
import pandas as pd
from datetime import datetime, timedelta


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
import sys


import pickle
import timeit
import datetime

start = timeit.default_timer()

def separatePMUs(X,y):
    
    """
    This function separates features and their corresponding labels for each PMU
    to make more events 
    """
    
    num_case=X.shape[0]
    num_pmu=X.shape[1]
    num_sample=X.shape[2]
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
        elif y[i]==4:
            for j in range(num_pmu):
                y2.append(4)       
        elif y[i]==5:
            for j in range(num_pmu):
                y2.append(5)    
    return X,np.array(y2)



def read_data(l):  

    # num = 1
    path = '2016-'
    p2 = open(path+ "tr_set.pickle","rb")
    X_train, y_train= pickle.load(p2)


    p2 = open(path+ "va_set.pickle","rb")
    X_test, y_test= pickle.load(p2)
    y_train = y_train[:,l]
    
    y_test = y_test[:,l]    
    # print('X_train.shape',X_train.shape)
    # print('y_train.shape',y_train.shape)
    # print('X_test.shape',X_test.shape)
    # print('y_test.shape',y_test.shape)    


    # only shuffle X_train part 
    # X_train, y_train =removePlanned(X_train, y_train)
    X_train, y_train=separatePMUs(X_train, y_train)
    # X_train, y_train = shuffle(X_train, y_train)   

    # X_test, y_test=removePlanned(X_test, y_test)
    X_test, y_test= separatePMUs(X_test,y_test)
 
    
    # print('X_train.shape',X_train.shape)
    # print('y_train.shape',y_train.shape)
    # print('X_test.shape',X_test.shape)
    # print('y_test.shape',y_test.shape)    
    
    
    return X_train,y_train,X_test,y_test    
    

    
def recover(df):
    dd = np.zeros(df.shape[0])
    dd = dd.astype("int8") 
    rr = df[df['L2']>0].index.tolist()
    dd[rr] = 1
    
    rr = df[df['L3']>0].index.tolist()
    dd[rr] = 2    


    rr = df[df['L4']>0].index.tolist()
    dd[rr] = 3    
    
    df['rev'] = dd
    return df
    
def evaluate(train_y,pred):
    
    # train_y = df['v'].values
    # pred = df[str(name)].values
    # a = train_y.shape
    matrix=confusion_matrix(train_y, pred)
    print(matrix)
    class_report=classification_report(train_y, pred)
    print(class_report)
    # print(str(name))    
    
def read_ll():  

    


    for i in range(1,4+1):
        df= pd.read_csv('label_'+str(i-1)+'.csv')
        list1.append(df['real'])

        
    dr = pd.DataFrame(list1).values.transpose()

    
    dr = pd.DataFrame(dr)
    dp = pd.DataFrame(dp)
    
    dr.columns = ['L1','L2','L3','L4']
    dp.columns = ['L1','L2','L3','L4']
    
    dr = recover(dr)
    dp = recover(dp)
    evaluate(dr['rev'],dp['rev'])

def read_ll():  

    list1 =[]
    for i in range(1,4+1):
        _,_,X_val, y_val= read_data(i-1)
        list1.append(y_val)
        

    string = '2016-muticlassbest_model_so_far'
    model = load_model(str(string)+'.h5')  
    y_pred=model.predict_classes(X_val) 
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns =['rev']
    print(y_pred.shape)
    y_val = pd.DataFrame(list1).values.transpose()
    y_val = pd.DataFrame(y_val)
    # print(y_val.shape)
    y_val.columns = ['L1','L2','L3','L4']
    
    
    y_val = recover(y_val)
    print(y_val.shape)
    evaluate(y_val['rev'],y_pred['rev'])
    
def main():
    read_ll()

    
    

if __name__ == '__main__':  
    # global best_accuracy 
    main()   
    
