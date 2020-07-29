

# import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import math
import pandas as pd
import time  



import pickle

import datetime


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
from sklearn import svm
from sklearn.metrics import accuracy_score
import timeit
import time  
from sklearn import metrics  
import pickle as pickle  
from sklearn.externals import joblib

from sklearn.model_selection import learning_curve, GridSearchCV  
from sklearn.svm import SVC    
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 


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
        

    return  np.array(X_new), np.array(y_new)


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
start = timeit.default_timer()


def read_data(l):  

    # num = 1
    path = 'S1+2'
    p2 = open(path+ "tr_set.pickle","rb")
    X_train, y_train= pickle.load(p2)


    p2 = open(path+ "va_set.pickle","rb")
    X_test, y_test= pickle.load(p2)
    y_train = y_train[:,l]
    
    y_test = y_test[:,l]    
    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)    


    # only shuffle X_train part 
    # X_train, y_train =removePlanned(X_train, y_train)
    X_train, y_train=separatePMUs(X_train, y_train)
    # X_train, y_train = shuffle(X_train, y_train)   

    # X_test, y_test=removePlanned(X_test, y_test)
    X_test, y_test= separatePMUs(X_test,y_test)
 
    
    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)    
    
    
    return X_train,y_train,X_test,y_test   

  
def read_data2():    
    tr_num = 50
    test_num =10
    channel_num = 6
    c_num =2
    X_train = np.random.random((tr_num, channel_num))
    y_train = np.random.randint(c_num, size=(tr_num, 1))
    # y_train = (np.random.randint(c_num, size=(100, 1)), num_classes=c_num)
    X_test = np.random.random((test_num, channel_num))
    y_test = np.random.randint(c_num, size=(test_num, 1))
    
    # X_train = X_train.astype(np.int32)
    # y_train = y_train.astype(np.int32)
    # X_test = X_test.astype(np.int32)
    # y_test = y_test.astype(np.int32)
    
    
    return X_train,y_train,X_test, y_test
    
    
    

  
def clf_svm():
    fname = 'svm-3'
    X_train,y_train,X_test, y_test = read_data(2)  
    model = SVC(kernel='rbf', probability=True)    
    param_grid = {'C': [350,400,450,500,550], 'gamma': [5e-3,6e-3,7e-3,8e-3]}    
    grid_search = GridSearchCV(model, param_grid, n_jobs = -1, verbose=1)    
    grid_search.fit(X_train, y_train)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
        
        
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    model.fit(X_train, y_train)           
    #保存Model(注:save文件夹要预先建立，否则会报错)
    with open(fname+ '.pickle', 'wb') as f:
        pickle.dump(model, f)

    #读取Model
    with open(fname+'.pickle', 'rb') as f:
        clf2 = pickle.load(f)
        #测试读取后的Model


    predict = model.predict(X_test) 
    train_out = model.predict(X_train)             

    print('training accucy:',accuracy_score(train_out,y_train))
    print('test accucy:',accuracy_score(predict,y_test))
    

    train_out2 = clf2.predict(X_train)
    predict2 = clf2.predict(X_test)
    
    print('training accucy:',accuracy_score(train_out2,y_train))
    print('test accucy:',accuracy_score(predict2,y_test))            

    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
        
def clf_DT():
    ac_value = 0

    X_train,y_train,X_test, y_test = read_data()  
    for i in range(1,3+1):
        dtr1=tree.DecisionTreeRegressor(max_depth=10+(i-1)*20)  

        dtr1.fit(X_train,y_train)

        predict2=dtr1.predict(X_test)
        train_out2 = dtr1.predict(X_train)
        train_out2=train_out2.astype(np.int32)
        predict2=predict2.astype(np.int32)
        
        # y_train = y_train.astype(np.int32)
        # y_test = y_test.astype(np.int32)
        # print(predict2)
        # print(y_test)
        acc = accuracy_score(predict2,y_test) 
        print('training accucy:',accuracy_score(train_out2,y_train))
        print('test accucy:',acc)                
        if ac_value< acc:
            ac_value = acc
            with open('clf.pickle', 'wb') as f:
                pickle.dump(dtr1, f)
            print('save done')
            
            #读取Model
    with open('clf.pickle', 'rb') as f:
        clf2 = pickle.load(f)
    predict2=clf2.predict(X_test)
    train_out2 = clf2.predict(X_train)
    train_out2=train_out2.astype(np.int32)
    predict2=predict2.astype(np.int32)        
    print('Final training accucy:',accuracy_score(train_out2,y_train))
    print('Final test accucy:',acc)      
    
def main(l):
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  

    # 
    # print('train_x.shape',train_x.shape)
    # print('train_y.shape',train_y.shape)
    # print('test_x.shape',test_x.shape)
    # print('test_y.shape',test_y.shape)
    
    

    
    train = 1
    if train == 1:
        clf_svm()
    

        
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))


    
    
if __name__ == '__main__':  

    main(0)

    # read_data(0)
