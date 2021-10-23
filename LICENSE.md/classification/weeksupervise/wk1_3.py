#This is using abstain one

import numpy as np
import math
import pandas as pd
import time  
from sklearn import metrics  
import pickle as pickle  
import timeit
import matplotlib.pyplot as plt
from scipy.linalg import svd 
# import time
import timeit
from random import randint
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# python wk1_test.py 2>&1 | tee bb.log
import random

    
def top():    
    rate = 0.3
    df1 = pd.read_csv("data/x1.csv")
    X_test = pd.read_csv("data/x2.csv")
    # y = df1.pop("label") 

    
    df1.pop("word")
    df1.pop("word2")
    df1.pop("season")
    df1.pop("No")
    
    ind1 = df1[df1["label"]==0].index.to_list()
    ind2 = df1[df1["label"]==1].index.to_list()
    ind3 = df1[df1["label"]==2].index.to_list()
    
    random.shuffle(ind1)
    random.shuffle(ind2)
    random.shuffle(ind3)
    
    a1 = int(rate*len(ind1))
    a2 = int(rate*len(ind2))
    a3 = int(rate*len(ind3))
    
    # print(ind1[:5])
    
    # random.shuffle(ind1)
    # print(ind1[:a1])
    # a = int(0.3*y.shape[0])
    
    sf1 = df1.iloc[ind1[:a1]] 
    sf2 = df1.iloc[ind2[:a2]] 
    sf3 = df1.iloc[ind3[:a3]] 

    sf4 = df1.iloc[ind1[a1:]] 
    sf5 = df1.iloc[ind2[a2:]] 
    sf6 = df1.iloc[ind3[a3:]] 
    

    X_train = pd.concat([sf1,sf2],axis=0)
    X_train = pd.concat([X_train,sf3],axis=0)
    X_train.sort_index(inplace=True)
    
    y_train = X_train.pop("label")
    
    X_val = pd.concat([sf4,sf5],axis=0)
    X_val = pd.concat([X_val,sf6],axis=0)
    X_val.sort_index(inplace=True)
    
    y_val = X_val.pop("label")
    
    # ll = y_val.tolist()
    # print(ll.count(1)/ll.count(0))
    # print(ll.count(2)/ll.count(0))


    # ll = y_train.tolist()
    # print(ll.count(1)/ll.count(0))
    # print(ll.count(2)/ll.count(0))
    

    
    return X_train,y_train,X_val,y_val
    

from sklearn.utils import shuffle


    
    
    
    

    

  
# Decision Tree Classifier  
def decision_tree_classifier(train_x, train_y):  
    from sklearn import tree  
    model = tree.DecisionTreeClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
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

def gradient_boosting_classifier(train_x, train_y):  
    from sklearn.ensemble import GradientBoostingClassifier  
    model = GradientBoostingClassifier()
    # model = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1500,max_depth=7, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10) 
    model.fit(train_x, train_y)  
    return model  
  
# SVM Classifier  
def svm_classifier(train_x, train_y):  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    model.fit(train_x, train_y)  
    return model  
  
# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=100)  
    model.fit(train_x, train_y)  
    return model  
  
def f2(df):
    dt = df.copy()
    dt = dt.replace(2,-1)
    # dt = dt.replace(1,2)
    # dt = dt.replace(0,1)
    # df = df.replace(1,2)    
    return dt

def f3(df):
    dt = df.copy()
    # dt = dt.replace(2,-1)
    dt = dt.replace(1,-1)

    return dt


    
def get_nosiy_all():
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  

    train_x, train_y,test_x, test_y = top()   
    
    # X_train = train_x.iloc[:,:18]
    # X_test = test_x.iloc[:,:18]
    
    # train_y = trans(train_y,0)
    # test_y = trans(test_y,0)
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
            # 'NB',
             'KNN', 
             'LR', 
            'RF', 
            # 'DT', 
            # 'SVM',

            'GBDT'
            ]  
            
            
        classifiers = {
        
                    # 'NB':naive_bayes_classifier,   
                      'KNN':knn_classifier,  
                       'LR':logistic_regression_classifier,  
                       'RF':random_forest_classifier,  
                       # 'DT':decision_tree_classifier,  
                      # 'SVM':svm_classifier,  
                    # 'SVMCV':svm_cross_validation,  
                     'GBDT':gradient_boosting_classifier  
        }  
          
        # print('reading training and testing data...')  
        # 
        # print('train_y.shape',y_test.shape)
        tr = pd.DataFrame(y_train)
        tr.columns = ['real']
        te = pd.DataFrame(y_test)
        te.columns = ['real']
        df  = y_test[:,np.newaxis]
        df2 = y_train[:,np.newaxis]
        
        list = []
        list.append('real')
        

        for classifier in test_classifiers:  
            print('******************* %s ********************' % classifier)  
            start_time = time.time()  
            model = classifiers[classifier](X_train, y_train)  
            # print('training took %fs!' % (time.time() - start_time))  
            predict = model.predict(X_test) 
            train_out = model.predict(X_train)  
            
            tr[classifier] = train_out
            te[classifier] = predict
            matrix=confusion_matrix(y_test, predict)
            print(matrix)
            class_report=classification_report(y_test, predict)
            print(class_report)
            
        # tr = f2(tr)
        # te = f2(te)
        tr.to_csv("data/wk_1_tr.csv",index = None)
        
        te.to_csv("data/wk_1_test.csv",index = None)
        # print("result save done")
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    

    
def get_nosiy_Line():
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  

    train_x, train_y,test_x, test_y = top()   
    
    X_train = train_x.iloc[:,:18]
    X_test = test_x.iloc[:,:18]
    
    # train_y = trans(train_y,0)
    # test_y = trans(test_y,0)
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
            # 'NB',
             'KNN', 
             'LR', 
            'RF', 
            # 'DT', 
            # 'SVM',

            'GBDT'
            ]  
            
            
        classifiers = {
        
                    # 'NB':naive_bayes_classifier,   
                      'KNN':knn_classifier,  
                       'LR':logistic_regression_classifier,  
                       'RF':random_forest_classifier,  
                       # 'DT':decision_tree_classifier,  
                      # 'SVM':svm_classifier,  
                    # 'SVMCV':svm_cross_validation,  
                     'GBDT':gradient_boosting_classifier  
        }  
          
        # print('reading training and testing data...')  
        # 
        # print('train_y.shape',y_test.shape)
        tr = pd.DataFrame(y_train)
        tr.columns = ['real']
        te = pd.DataFrame(y_test)
        te.columns = ['real']
        df  = y_test[:,np.newaxis]
        df2 = y_train[:,np.newaxis]
        
        list = []
        list.append('real')
        

        for classifier in test_classifiers:  
            print('******************* %s ********************' % classifier)  
            start_time = time.time()  
            model = classifiers[classifier](X_train, y_train)  
            # print('training took %fs!' % (time.time() - start_time))  
            predict = model.predict(X_test) 
            train_out = model.predict(X_train)  
            
            tr["v_"+classifier] = train_out
            te["v_"+classifier] = predict
            # matrix=confusion_matrix(y_test, predict)
            # print(matrix)
            # class_report=classification_report(y_test, predict)
            # print(class_report)
            
        tr = f2(tr)
        te = f2(te)
        tr.to_csv("data/wk_1_tr2.csv",index = None)
        
        te.to_csv("data/wk_1_test2.csv",index = None)
        # print("result save done")
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
def get_nosiy_Trans():
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  

    train_x, train_y,test_x, test_y = top()   
    X_train = train_x.iloc[:,18:36]
    X_test = test_x.iloc[:,18:36]
    
    # train_y = trans(train_y,1)
    # test_y = trans(test_y,1)
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
            # 'NB',
             'KNN', 
             'LR', 
            'RF', 
            # 'DT', 
            # 'SVM',

            'GBDT'
            ]  
            
            
        classifiers = {
        
                    # 'NB':naive_bayes_classifier,   
                      'KNN':knn_classifier,  
                       'LR':logistic_regression_classifier,  
                       'RF':random_forest_classifier,  
                       # 'DT':decision_tree_classifier,  
                      # 'SVM':svm_classifier,  
                    # 'SVMCV':svm_cross_validation,  
                     'GBDT':gradient_boosting_classifier  
        }  
          
        # print('reading training and testing data...')  
        # 
        # print('train_y.shape',y_test.shape)
        tr = pd.DataFrame(y_train)
        tr.columns = ['real']
        te = pd.DataFrame(y_test)
        te.columns = ['real']
        df  = y_test[:,np.newaxis]
        df2 = y_train[:,np.newaxis]
        
        list = []
        list.append('real')
        

        for classifier in test_classifiers:  
            print('******************* %s ********************' % classifier)  
            start_time = time.time()  
            model = classifiers[classifier](X_train, y_train)  
            # print('training took %fs!' % (time.time() - start_time))  
            predict = model.predict(X_test) 
            train_out = model.predict(X_train)  
            
            tr["i_"+classifier] = train_out
            te["i_"+classifier] = predict
            # matrix=confusion_matrix(y_test, predict)
            # print(matrix)
            # class_report=classification_report(y_test, predict)
            # print(class_report)
        tr = f2(tr)
        te = f2(te)
        tr.to_csv("data/wk_1_tr3.csv",index = None)
        
        te.to_csv("data/wk_1_test3.csv",index = None)
        # print("result save done")
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))

def get_nosiy_Freq():
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  

    train_x, train_y,test_x, test_y = top()   
    X_train = train_x.iloc[:,36:]
    X_test = test_x.iloc[:,36:]    
    
    # train_y = trans(train_y,2)
    # test_y = trans(test_y,2)
    # X_train = train_x
    # X_test = test_x
    # X_train = train_x.iloc[:,36:]
    # X_test = test_x.iloc[:,36:]
    y_test = test_y
    y_train = train_y
 

    train = 1
    if train == 1:
        thresh = 0.5  
        model_save_file = None  
        model_save = {}  
       
       
        test_classifiers = [
            # 'NB',
             'KNN', 
             'LR', 
            'RF', 
            # 'DT', 
            # 'SVM',

            'GBDT'
            ]  
            
            
        classifiers = {
        
                    # 'NB':naive_bayes_classifier,   
                      'KNN':knn_classifier,  
                       'LR':logistic_regression_classifier,  
                       'RF':random_forest_classifier,  
                       # 'DT':decision_tree_classifier,  
                      # 'SVM':svm_classifier,  
                    # 'SVMCV':svm_cross_validation,  
                     'GBDT':gradient_boosting_classifier  
        }  
          
        # print('reading training and testing data...')  
        # 
        # print('train_y.shape',y_test.shape)
        tr = pd.DataFrame(y_train)
        tr.columns = ['real']
        te = pd.DataFrame(y_test)
        te.columns = ['real']
        df  = y_test[:,np.newaxis]
        df2 = y_train[:,np.newaxis]
        
        list = []
        list.append('real')
        

        for classifier in test_classifiers:  
            print('******************* %s ********************' % classifier)  
            start_time = time.time()  
            model = classifiers[classifier](X_train, y_train)  
            # print('training took %fs!' % (time.time() - start_time))  
            predict = model.predict(X_test) 
            train_out = model.predict(X_train)  
            
            tr["r_"+classifier] = train_out
            te["r_"+classifier] = predict
            # matrix=confusion_matrix(y_test, predict)
            # print(matrix)
            # class_report=classification_report(y_test, predict)
            # print(class_report)
        tr = f3(tr)
        te = f3(te)            
        tr.to_csv("data/wk_1_tr4.csv",index = None)
        
        te.to_csv("data/wk_1_test4.csv",index = None)
        # print("result save done")
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))

def get_real():
    train_x, train_y,test_x, test_y = top() 
    # train_y[train_y==2]=3
    # train_y[train_y==1]=2
    # train_y[train_y==0]=1
    
    # test_y[test_y==2]=3
    # test_y[test_y==1]=2
    # test_y[test_y==0]=1
    
    tr = pd.DataFrame(train_y)
    te = pd.DataFrame(test_y)
    tr.to_csv("data/wk_1_tr5.csv",index = None)
    te.to_csv("data/wk_1_test5.csv",index = None)
    
def get_semi_label():
    train_x, train_y,test_x, test_y = top()   
    y = np.zeros(test_y.shape[0])
    for i in range(y.shape[0]):
        y[i] = -1
    
    # print(train_x.shape)
    # print(train_y.shape)
    
    # print(test_x.shape)
    # print(y.shape)
    
    train_x = np.concatenate((train_x, test_x), axis=0)
    train_y = np.concatenate((train_y, y), axis=0)
    
    # print(train_x.shape)
    # print(train_y.shape)
    
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier  
    from sklearn.semi_supervised import LabelSpreading
    from sklearn.semi_supervised import SelfTrainingClassifier
    base_classifier = RandomForestClassifier(n_estimators=100)  
    semi_res1 = SelfTrainingClassifier(base_classifier).fit(train_x, train_y).predict(test_x)
    semi_res2 =  LabelSpreading().fit(train_x, train_y).predict(test_x)
    print(semi_res1.shape)
    print(semi_res2.shape)
    
    df = pd.DataFrame(semi_res1)
    df.columns = ['SelfTrain']
    # df["LabelSpread"] = semi_res2
    df.to_csv("semi_res.csv",index = 0)
    
def main():
    s1 = timeit.default_timer()

    get_nosiy_Line()
    get_nosiy_Trans()
    get_nosiy_Freq()
    get_real()
    get_semi_label()
    s2 = timeit.default_timer()
    print('Time:(min) ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

