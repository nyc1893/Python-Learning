

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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import sys
sys.path.append("..")
import get_label
    

  
def clf_svm():
    fname = 'opt/svm-1'
    X_train,y_train,X_test, y_test = cc()  
    model = SVC(kernel='rbf', probability=True)    
    param_grid = {'C': [1e-3,1e-2,1e-1,1, 1e2, 1e3, 1e4], 'gamma': [1e-3,1e-2,1e-4]}    
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
        

def test_svm():
    fname = 'svm-1'
    X_train,y_train,X_test, y_test = cc()  

    #读取Model
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


    
def main(l):
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  

    # 
    # print('train_x.shape',train_x.shape)
    # print('train_y.shape',train_y.shape)
    # print('test_x.shape',test_x.shape)
    # print('test_y.shape',test_y.shape)
    
    

    
    train = 0
    if train == 1:
        clf_svm()
    test_svm()

        
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))



    
def clf_svm2():
    # fname = 'svm-3'
    X_train,y_train,X_test, y_test = read_data(2)  
    
    AR_score = make_scorer(AR, greater_is_better=True, needs_proba = True)
    scoring = {'AR': AR_score}
    param_grid = {'C': [350,400], 'gamma': [5e-3,6e-3]}    
    model = SVC(kernel='rbf', probability=True)   
    grid_search = GridSearchCV(model, param_grid ,scoring = scoring, n_jobs = -1, verbose=1)   
    
  
    grid_search.fit(X_train, y_train)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)        
    

        
    
if __name__ == '__main__':  

    main(0)
    
    # clf_svm2()
    # read_data(0)
