

# import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import math
import pandas as pd
import time  
from sklearn import metrics  
import pickle as pickle  
import pandas as pd


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


import timeit
import time  
from sklearn import metrics  
import pickle as pickle  
import pandas as pd


from rno_fun import data_lowfreq

    
    

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
  

  

  

      
def run():
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  

    train_x, train_y,test_x, test_y =readdata()   

    
 
    X_train = train_x
    X_test = test_x
    
    y_test = test_y
    y_train = train_y
 

    train = 1
    if train == 1:
        thresh = 0.5  
        model_save_file = None  
        model_save = {}  
       
        test_classifiers = [

             'KNN', 
             'LR', 
            'RF', 
            # 'DT', 
            'SVM',

            'GBDT'
            ]  
            
            
        classifiers = {
        
                    # 'NB':naive_bayes_classifier,   
                      'KNN':knn_classifier,  
                       'LR':logistic_regression_classifier,  
                       'RF':random_forest_classifier,  
                       'DT':decision_tree_classifier,  
                      'SVM':svm_classifier,  
                    # 'SVMCV':svm_cross_validation,  
                     'GBDT':gradient_boosting_classifier  
        }  
          
        print('reading training and testing data...')  
        # 
        print('train_y.shape',y_test.shape)
        df  = y_test[:,np.newaxis]
        df2 = y_train[:,np.newaxis]
        
        list = []
        list.append('real')
        for classifier in test_classifiers:  
            print('******************* %s ********************' % classifier)  
            start_time = time.time()  
            model = classifiers[classifier](train_x, train_y)  
            print('training took %fs!' % (time.time() - start_time))  
            predict = model.predict(X_test) 
            train_out = model.predict(X_train)             
            # np.save(classifier+"_pred.npy",predict)
            
            predict = predict[:,np.newaxis]
            train_out = train_out[:,np.newaxis]
            
            print(df.shape)
            print(predict.shape)
            print(df2.shape)
            print(train_out.shape)           
            df = np.concatenate((df,predict),axis=1)
            df2 = np.concatenate((df2,train_out),axis=1)
            
            list.append(str(classifier))
            # precision = metrics.precision_score(test_y, predict)  
            # recall = metrics.recall_score(test_y, predict)  
            # print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))  
            # accuracy = metrics.accuracy_score(test_y, predict)  
            # print('accuracy: %.2f%%' % (100 * accuracy))   
            # list.append(predict)
            # if model_save_file != None:  
                # model_save[classifier] = model  

        df = pd.DataFrame(df,columns=list)
        df2 = pd.DataFrame(df2,columns=list)
        
        df.to_csv('Ltest_34.csv',index =None)
        df2.to_csv('Ltrain_12.csv',index =None)
        

        print('save done!')

    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
    
def rd1(i):
    path = 'data/'
    X_train =  pd.read_csv(path +'D'+str(i)+'_Ltr.csv')
    X_train = X_train.rename(columns={'21':'label'})
    y_train = X_train.pop('label')
    y_train = y_train.astype('int') 
    # print(X_train.head())
    
    X_test =  pd.read_csv(path +'D'+str(i)+'_Ltest.csv')
    X_test = X_test.rename(columns={'21':'label'})
    y_test = X_test.pop('label')
    y_test = y_test.astype('int') 
    return  X_train.values,y_train.values,X_test.values,y_test.values
    
def readdata():
    X_train,y_train, X_val, y_val= rd1(1)
    for i in range(2,7):
        X_train2,y_train2, X_val2, y_val2= rd1(i)
        X_train = np.concatenate((X_train,X_train2), axis=0)
        y_train = np.concatenate((y_train,y_train2), axis=0)
        X_val = np.concatenate((X_val,X_val2), axis=0)
        y_val = np.concatenate((y_val,y_val2), axis=0)
        
    print('unique-ytran',np.unique(y_train))
    print('X_train.shape',X_train.shape) 
    print('y_train.shape',y_train.shape) 
    print('X_val.shape',X_val.shape) 
    print('y_val.shape',y_val.shape)     
    return X_train,y_train, X_val, y_val
    
def main():
    run()

if __name__ == '__main__':  

    main()

