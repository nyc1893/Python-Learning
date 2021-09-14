

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


def div_half():
    path = "data/"
    df = pd.read_csv(path+ "X_train.csv")
    df1 = df[df.index%2==0]
    df2 = df[df.index%2==1]

    df1.to_csv(path+"X_r.csv",index = 0)
    df2.to_csv(path+"X_n.csv",index = 0)
    print("save done")

def top():
    path = "data/"
    df1 = pd.read_csv(path+"X_r.csv")
    df2 = pd.read_csv(path+"X_n.csv")   
    train_y = df1.pop("label") 
    test_y = df2.pop("label")
    
    df1.pop("word")
    df1.pop("word2")
    df1.pop("season")
    df1.pop("No")
    
    df2.pop("word")
    df2.pop("word2")
    df2.pop("season")
    df2.pop("No")   
    
    train_x = df1
    test_x = df2
    return train_x, train_y,test_x, test_y
    
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
  
def naive_bayes_classifier(train_x, train_y): 
  from sklearn.naive_bayes import MultinomialNB 
  model = MultinomialNB(alpha=0.01) 
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
  
def LF_1(df):  
    # df = train_x.copy()
    df["LF1"] = 0
    df.loc[(df.r1>1),"LF1" ] = 2
    return df
    
def LF_2(df):  
    # df = train_x.copy()
    df["LF2"] = 0
    df.loc[(df.r3>1),"LF2" ] = 2
    return df

def LF_3(df):  
    # df = train_x.copy()
    df["LF3"] = 0
    df.loc[(df.mean_i_up<10),"LF3" ] = 1
    return df
    

def get_nosiy_label():
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  

    train_x, train_y,test_x, test_y = top()   

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
            # 'RF', 
            'DT', 
            'SVM',

            # 'GBDT'
            ]  
            
            
        classifiers = {
        
                    # 'NB':naive_bayes_classifier,   
                      'KNN':knn_classifier,  
                       'LR':logistic_regression_classifier,  
                       # 'RF':random_forest_classifier,  
                       'DT':decision_tree_classifier,  
                      'SVM':svm_classifier,  
                    # 'SVMCV':svm_cross_validation,  
                     # 'GBDT':gradient_boosting_classifier  
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
        # X_train= LF_1(X_train)
        # X_train= LF_2(X_train)
        # X_train= LF_3(X_train)
        
        # X_test= LF_1(X_test)
        # X_test= LF_2(X_test)
        # X_test= LF_3(X_test)
        
        # tr["LF1"] = X_train["LF1"]
        # tr["LF2"] = X_train["LF2"]
        # tr["LF3"] = X_train["LF3"]
        
        # te["LF1"] = X_test["LF1"]
        # te["LF2"] = X_test["LF2"]
        # te["LF3"] = X_test["LF3"]
        
        # tr.to_csv("wk_1_tr.csv",index = None)
        
        # te.to_csv("wk_1_test.csv",index = None)
        # print("result save done")
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    

    
    
    
def main():
    s1 = timeit.default_timer()  
    # div_half()
    get_nosiy_label()
    s2 = timeit.default_timer()
    print('Time:(min) ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

