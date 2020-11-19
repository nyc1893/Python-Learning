
# This is for Feq CWT^2 use
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import pandas as pd
import pywt
from sklearn.decomposition import PCA

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

import os
import sys
start = timeit.default_timer()

def TR(X_train):
    num = 20
    pca_2 = PCA(n_components=num)
    X_train = pca_2.fit_transform(X_train)
    return X_train
    



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
            X_new.append(X[i,:])
    
        elif y[i]==1:
            y_new.append(1)
            X_new.append(X[i,:])
    
            
        elif y[i]==2:
            y_new.append(2)
            X_new.append(X[i,:])
        
        elif y[i]==3:
            y_new.append(3)
            X_new.append(X[i,:])
            
        elif y[i]==4:
            y_new.append(0)
            X_new.append(X[i,:])
        
        elif y[i]==5:
            y_new.append(1)
            X_new.append(X[i,:])
            
        elif y[i]==6:
            y_new.append(6)
            X_new.append(X[i,:])
        
        elif y[i]==7:
            y_new.append(7)
            X_new.append(X[i,:])                      
        

    return  np.array(X_new), np.array(y_new)
    

def get_time(String):
    year = String.split("_")[0]
    day = String.split("_")[2]
    num = int(String.split("_")[3])

    if String.split("_")[1] == 'Jan':
        month = 1
    elif String.split("_")[1] == 'Feb':
        month = 2
    elif String.split("_")[1] == 'Mar':
        month = 3
    elif String.split("_")[1] == 'Apr':
        month = 4
    elif String.split("_")[1] == 'May':
        month = 5 
    elif String.split("_")[1] == 'Jun':
        month = 6
    elif String.split("_")[1] == 'Jul':
        month = 7  
        
    elif String.split("_")[1] == 'Aug':
        month = 8
    elif String.split("_")[1] == 'Sep':
        month = 9
    elif String.split("_")[1] == 'Oct':
        month = 10
    elif String.split("_")[1] == 'Nov':
        month = 11 
    elif String.split("_")[1] == 'Dec':
        month = 12
           
    # print(year,month,day,num)
    String = str(month) +'/'+day+'/'+ year
    return num,String

def deal_label(y_test):
    path1 = '../y_label/'
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
    
    
    
def get_class(file_name,String):    

    data = pd.read_csv(file_name)
    data['Start'] = pd.to_datetime(data['Start'])
    data = data.set_index('Start')
    num, str =get_time(String)
    dt = data[str]
    # print(dt.iloc[num,2])
    return dt.iloc[num,2]



def rd2(i):
    path1 = '../set/'


    p1 = open(path1 +'X'+str(i)+'_single_vpa3.pickle',"rb")
    p3  = open(path1 +'y'+str(i)+'_single_vpa3.pickle',"rb")    
    
    pk3 = pickle.load(p3)
    print(pk3.shape)
    pk3 = deal_label(pk3)
    print(pk3.shape)
    # pk3 = pk3[:,1]   
    # pk3 = pk3.astype(np.int32)          
    pk1 = pickle.load(p1)

    fps=60


    path2 = 'index_single3/'
    tr=np.load(path2 +'tr_' +str(i)+'.npy')
    val=np.load(path2 +'val_' +str(i)+'.npy')
    tr=tr.tolist()  
    val=val.tolist() 

    

    X_train = pk1[tr]
    y_train = pk3[tr]
    X_val = pk1[val]
    y_val = pk3[val]
    
    X_train,y_train=rm2(X_train,y_train)
    X_val,y_val = rm2(X_val,y_val)
    print(X_train.shape) 
    print(X_val.shape) 
    print(y_train.shape) 
    print(y_val.shape) 
    """
    """
    return X_train, X_val, y_train, y_val    

    
def DWT(X_train):


    (ca1, cd1) = pywt.dwt(X_train, 'db2')
    (ca2, cd2) = pywt.dwt(ca1, 'db2')
    (ca3, cd3) = pywt.dwt(ca2, 'db2')
    (ca4, cd4) = pywt.dwt(ca3, 'db2')
    (ca5, cd5) = pywt.dwt(ca4, 'db2')
    (ca6, cd6) = pywt.dwt(ca5, 'db2')
    
    num = 20
    pca_2 = PCA(n_components=num)
    
    ca6 = pca_2.fit_transform(ca6)
    ca5 = pca_2.fit_transform(ca5)    
    
    ca4 = pca_2.fit_transform(ca4)
    ca3 = pca_2.fit_transform(ca3)
    ca2 = pca_2.fit_transform(ca2)
    ca1 = pca_2.fit_transform(ca1)
    
    cd6 = pca_2.fit_transform(cd6)
    cd5 = pca_2.fit_transform(cd5)    
    cd4 = pca_2.fit_transform(cd4)
    cd3 = pca_2.fit_transform(cd3)
    cd2 = pca_2.fit_transform(cd2)
    cd1 = pca_2.fit_transform(cd1)    
    # print('cd1.shape',cd1.shape) 
    # print('cd2.shape',cd2.shape) 
    # print('cd3.shape',cd3.shape) 
    # print('cd4.shape',cd4.shape) 
    # print('cd5.shape',cd5.shape) 
    # print('cd6.shape',cd6.shape) 
    
    # print('ca1.shape',ca1.shape) 
    # print('ca2.shape',ca2.shape) 
    # print('ca3.shape',ca3.shape) 
    # print('ca4.shape',ca4.shape)    
    # print('ca5.shape',ca5.shape) 
    # print('ca6.shape',ca6.shape)     
    return ca6,ca5, ca4,ca3,ca2,ca1,cd6,cd5,cd4,cd3,cd2,cd1
    

    
    

# va_a
def dprocL(i):
    X_train, X_val, y_train, y_val = rd2(i)
    # for j in range(2,6+1):
        # X_train2, X_val2, y_train2, y_val2 = rd2(j,i)
        # X_train = np.concatenate((X_train,X_train2))
        # y_train = np.concatenate((y_train,y_train2))
        # X_val = np.concatenate((X_val,X_val2))
        # y_val = np.concatenate((y_val,y_val2))        
    
    # X_train,  y_train = rm2 (X_train,  y_train)
    # X_val,  y_val = rm2 (X_val,  y_val)
    # X_train,  y_train = separatePMUs (X_train,  y_train)
    # X_val,  y_val = separatePMUs (X_val,  y_val)

    num= X_train.shape[0]

    p1 = np.concatenate((X_train,X_val))
    p1 = p1.reshape(p1.shape[0],-1)
    d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11 = DWT(p1)
    
    p3 = locals()['d'+str(0)] 
    X_train = p3[:num]
    X_val = p3[num:]    
        
    # for i in range(1,5):
        # p3 = locals()['d'+str(i)] 
        # X_train2 = p3[:num]
        # X_val2 = p3[num:]

        # X_train = np.concatenate((X_train,X_train2), axis=1)
        # X_val = np.concatenate((X_val,X_val2), axis=1)

    
    print('X_train.shape',X_train.shape) 
    print('y_train.shape',y_train.shape) 
    print('X_val.shape',X_val.shape) 
    print('y_val.shape',y_val.shape) 
    return  X_train,y_train, X_val, y_val
    

    
    
def data_pack2():
    X_train,y_train, X_val, y_val= dprocL(1)
    

    X_train2,y_train2, X_val2, y_val2= dprocL(2)
    X_train = np.concatenate((X_train,X_train2), axis=0)
    y_train = np.concatenate((y_train,y_train2), axis=0)
    X_val = np.concatenate((X_val,X_val2), axis=0)
    y_val = np.concatenate((y_val,y_val2), axis=0)
    
    X_train2,y_train2, X_val2, y_val2= dprocL(3)
    X_train = np.concatenate((X_train,X_train2), axis=0)
    y_train = np.concatenate((y_train,y_train2), axis=0)
    X_val = np.concatenate((X_val,X_val2), axis=0)
    y_val = np.concatenate((y_val,y_val2), axis=0)
    
    X_train2,y_train2, X_val2, y_val2= dprocL(4)
    X_train = np.concatenate((X_train,X_train2), axis=0)
    y_train = np.concatenate((y_train,y_train2), axis=0)
    X_val = np.concatenate((X_val,X_val2), axis=0)
    y_val = np.concatenate((y_val,y_val2), axis=0)

    X_train2,y_train2, X_val2, y_val2= dprocL(5)
    X_train = np.concatenate((X_train,X_train2), axis=0)
    y_train = np.concatenate((y_train,y_train2), axis=0)
    X_val = np.concatenate((X_val,X_val2), axis=0)
    y_val = np.concatenate((y_val,y_val2), axis=0)

    X_train2,y_train2, X_val2, y_val2= dprocL(6)
    X_train = np.concatenate((X_train,X_train2), axis=0)
    y_train = np.concatenate((y_train,y_train2), axis=0)
    X_val = np.concatenate((X_val,X_val2), axis=0)
    y_val = np.concatenate((y_val,y_val2), axis=0)    
    
    print('unique-ytran',np.unique(y_train))
    print('X_train.shape',X_train.shape) 
    print('y_train.shape',y_train.shape) 
    print('X_val.shape',X_val.shape) 
    print('y_val.shape',y_val.shape) 
    return  X_train,y_train, X_val, y_val
    
def clf_DT():
    ac_value = 0

    X_train,y_train,X_test, y_test = data_pack2()  
    for i in range(1,8+1):
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
        # print('training accucy:',accuracy_score(train_out2,y_train))
        # print('test accucy:',acc)                
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
    matrix=confusion_matrix(predict2,y_test)

    print(matrix)
    class_report=classification_report(predict2,y_test)
    print(class_report)
    
    
def clf_svm():
    fname = 'svm-3'
    X_train,y_train,X_test, y_test = data_pack2()  
    model = SVC(kernel='rbf', probability=True)    
    param_grid = {'C': [1e-1,1e-2,1e-3,1,1e1,1e2,1e3], 'gamma': [1e-3,1e-2,1e-1]}    
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

    with open(fname+'.pickle', 'rb') as f:
        clf2 = pickle.load(f)

    print('training accucy:',accuracy_score(train_out2,y_train))
    print('test accucy:',accuracy_score(predict2,y_test))            

    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    



    train_out2 = clf2.predict(X_train)
    predict2 = clf2.predict(X_test)
    
    print('training accucy:',accuracy_score(train_out2,y_train))
    print('test accucy:',accuracy_score(predict2,y_test))     
    
    matrix=confusion_matrix(predict2,y_test)
    print(fname)
    print(matrix)
    class_report=classification_report(predict2,y_test)
    print(class_report)
    
def clf_svm2():
    fname = 'svm-3'
    X_train,y_train,X_test, y_test = data_pack2()  
    model = SVC(kernel='linear', probability=True)    
    param_grid = {'C': [1e-1,1e-2,1e-3,1,1e1,1e2,1e3]}    
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

    

    train_out2 = clf2.predict(X_train)
    predict2 = clf2.predict(X_test)
    
    print('training accucy:',accuracy_score(train_out2,y_train))
    print('test accucy:',accuracy_score(predict2,y_test))            

    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    

    with open(fname+'.pickle', 'rb') as f:
        clf2 = pickle.load(f)

    train_out2 = clf2.predict(X_train)
    predict2 = clf2.predict(X_test)
    
    print('training accucy:',accuracy_score(train_out2,y_train))
    print('test accucy:',accuracy_score(predict2,y_test))     
    
    matrix=confusion_matrix(predict2,y_test)
    print(fname)
    print(matrix)
    class_report=classification_report(predict2,y_test)
    print(class_report)
    
# KNN Classifier  
def knn_classifier():  
    from sklearn.neighbors import KNeighborsClassifier  
    best_k = 0 
    best_p = 0 
    best_score =0
    fname = 'knn-1'
    X_train,y_train,X_test, y_test = data_pack2()  
    for k in range(1,20):
        for p in range(1,10):
            model = KNeighborsClassifier(n_neighbors = k, weights = 'distance', p = p)
            model.fit(X_train,y_train)  
            score = model.score(X_test, y_test)
            if score > best_score:
                best_k = k
                best_p = p
                best_score = score
                
    model = KNeighborsClassifier(n_neighbors = best_k, weights = 'distance', p = best_p)
    model.fit(X_train,y_train)      
    

    
    with open(fname+ '.pickle', 'wb') as f:
        pickle.dump(model, f)

    #读取Model
    with open(fname+'.pickle', 'rb') as f:
        clf2 = pickle.load(f)
        
    train_out2 = clf2.predict(X_train)
    predict2 = clf2.predict(X_test)   
    
    print('training accucy:',accuracy_score(train_out2,y_train))
    print('test accucy:',accuracy_score(predict2,y_test))     
    print('best_k',best_k)
    print('best_p',best_p)
    matrix=confusion_matrix(predict2,y_test)

    print(matrix)
    class_report=classification_report(predict2,y_test)
    print(class_report)
 # Random Forest Classifier  
def rf_para():  
    from sklearn.ensemble import RandomForestClassifier  
    fname = 'rf-2'
    X_train,y_train,X_test, y_test = data_pack2()  
    
    param_test1 = {"n_estimators":range(1,201,10)}
    # gsearch1 = GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_test1,n_jobs = -1,cv=5)
    
    param_test2 = {"max_features":range(1,11,1)}
    gsearch1 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=101),
                param_grid=param_test2,n_jobs = -1,cv=5)    
    
    gsearch1.fit(X_train,y_train)

    # print(gsearch1.grid_scores_)
    print(gsearch1.best_params_)
    print("best accuracy:%f" % gsearch1.best_score_)
    
# Random Forest Classifier  
def rf():  
    from sklearn.ensemble import RandomForestClassifier  
    fname = 'rf-2'
    X_train,y_train,X_test, y_test = data_pack2()  
    
    model = RandomForestClassifier(n_estimators=101, max_features=1)  
    model.fit(X_train,y_train)  
    with open(fname+ '.pickle', 'wb') as f:
        pickle.dump(model, f)

    #读取Model
    with open(fname+'.pickle', 'rb') as f:
        clf2 = pickle.load(f)
        
    train_out2 = clf2.predict(X_train)
    predict2 = clf2.predict(X_test)   
    
    print('training accucy:',accuracy_score(train_out2,y_train))
    print('test accucy:',accuracy_score(predict2,y_test))     

    matrix=confusion_matrix(predict2,y_test)

    print(matrix)
    class_report=classification_report(predict2,y_test)
    print(class_report)    
    
def main():
    s1 = timeit.default_timer()  
    rf()
    # rf_para()
    # knn_classifier()
    # clf_svm()
    # clf_DT()
    # dprocL(1)
    # data_pack2()
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
if __name__ == '__main__':  
    # k = int(sys.argv[1])
    main()
