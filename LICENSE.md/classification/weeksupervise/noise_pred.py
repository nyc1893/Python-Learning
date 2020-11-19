

# import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import math
import pandas as pd
import time  
from sklearn import metrics  
import pickle as pickle  
import pandas as pd

import sys
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

from sklearn.model_selection import learning_curve, GridSearchCV  
from sklearn.svm import SVC    
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import timeit
import time  
from sklearn import metrics  
import pickle as pickle  
import pandas as pd


from get_label import readdata

    
    

# KNN Classifier  
def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
  
# Logistic Regression Classifier  
def logistic_regression_classifier(train_x, train_y):  
    from sklearn.linear_model import LogisticRegression  
    model = LogisticRegression(penalty='l2')  
    model.fit(train_x, train_y)  
    return model  
  
  
# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=100)  
    model.fit(train_x, train_y)  
    return model  
  
  
# Decision Tree Classifier  
def decision_tree_classifier(train_x, train_y):  
    from sklearn import tree  
    model = tree.DecisionTreeClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
  
# GBDT(Gradient Boosting Decision Tree) Classifier  
def gradient_boosting_classifier(train_x, train_y):  
    from sklearn.ensemble import GradientBoostingClassifier  
    model = GradientBoostingClassifier(n_estimators=200)  
    model.fit(train_x, train_y)  
    return model  
  
  
# SVM Classifier  
def svm_classifier(train_x, train_y):  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    model.fit(train_x, train_y)  
    return model  
  
def datapack(k):

    list = ['real',	'KNN',	'LR','RF','SVM','GBDT','SNO','Maj']
    df = pd.read_csv('Ltest_34.csv')
    print(df.head())
    X_train,y_train, X_val, y_val = readdata()

    y_train2 = df[str(list[k])]


    X_train = np.concatenate((X_train,X_val), axis=0)
    y_train = np.concatenate((y_train,y_train2), axis=0)
    X_test,y_test = read_data2()
    print('unique-ytran',np.unique(y_train))
    print('X_train.shape',X_train.shape) 
    print('y_train.shape',y_train.shape)    

    print('X_test.shape',X_test.shape) 
    print('y_test.shape',y_test.shape)   
    
    print(str(list[k]))
    return  X_train,y_train,X_test,y_test    
    
def rd2(i):
    path = 'data/'

    X_test =  pd.read_csv(path +'D'+str(i)+'_ytest.csv')
    X_test = X_test.rename(columns={'21':'label'})
    y_test = X_test.pop('label')
    y_test = y_test.astype('int') 
    return  X_test.values,y_test.values

def read_data2():

    X_train,y_train= rd2(1)
    for i in range(2,7):
        X_train2,y_train2= rd2(i)
        X_train = np.concatenate((X_train,X_train2), axis=0)
        y_train = np.concatenate((y_train,y_train2), axis=0)
    # print('unique-ytran',np.unique(y_train))
    # print('X_test.shape',X_train.shape) 
    # print('y_test.shape',y_train.shape)          
    return  X_train,y_train  
  
def clf_svm(k):
    fname = 'svm-3'
    X_train, y_train =read_data1(k)   
    X_test, y_test= read_data3()    
    
    
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

def clf_svm2(k):
    fname = 'svm-3'
    X_train,y_train,X_test, y_test = datapack(k)  
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


def main(k):
    s1 = timeit.default_timer()  
    clf_svm2(k)
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
if __name__ == '__main__':  
    k = int(sys.argv[1])
    main(k)

