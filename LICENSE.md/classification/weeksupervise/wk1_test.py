

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


def fix_data():
    df1,X_test=filter_data()
    df1.to_csv("data/x1.csv",index =0)
    X_test.to_csv("data/x2.csv",index =0)
    
    
def top():    

    df1 = pd.read_csv("data/x1.csv")
    X_test = pd.read_csv("data/x2.csv")
    y = df1.pop("label") 

    
    df1.pop("word")
    df1.pop("word2")
    df1.pop("season")
    df1.pop("No")
    

    
    a = int(0.33*y.shape[0])
    
    X_train = df1.iloc[:a] 
    y_train = y.iloc[:a]   
    X_val = df1.iloc[a:] 
    y_val = y.iloc[a:]   

    print(X_train.shape)
    print(X_val.shape)
    
    return X_train,y_train,X_val,y_val
    

from sklearn.utils import shuffle

def filter_data():
    path = "data/"
    df2 = pd.read_csv(path+"X_train.csv")
    df4 = df2[df2["label"] == 0]
    df5 = df2[df2["label"] == 2]
    
    df1 = df2[df2["label"] == 1]
    print(df1.shape)
    # df3 = df1[(df1["max_v_dup"] > 0.01) | (df1["max_v_ddn"] >0.005)
    # | (df1["max_v_adn"] > 2)| (df1["max_v_aup"] > 0.07)]
    
    # df4 = df4[(df4["max_v_dup"] < 0.01) | (df4["max_v_ddn"] <0.005)
    # | (df4["max_v_adn"] <2)| (df4["max_v_aup"] < 0.07)]
    df3 = df1
    df  = pd.concat([df4, df5])
    df  = pd.concat([df, df3])   
    df = shuffle(df)
    X_train = df
    df2 = pd.read_csv(path+"X_val.csv")
    df4 = df2[df2["label"] == 0]
    df5 = df2[df2["label"] == 2]
    
    df1 = df2[df2["label"] == 1]
    print(df1.shape)
    # df3 = df1[(df1["max_v_dup"] > 0.01) | (df1["max_v_ddn"] >0.005)
    # | (df1["max_v_adn"] > 2)| (df1["max_v_aup"] > 0.07)]

    # df4 = df4[(df4["max_v_dup"] < 0.01) | (df4["max_v_ddn"] <0.005)
    # | (df4["max_v_adn"] <2)| (df4["max_v_aup"] < 0.07)]
    
    df3 = df1
    df  = pd.concat([df4, df5])
    df  = pd.concat([df, df3])   
    df = shuffle(df)
    X_test = df
    
    return X_train,X_test
    
    
    
    
def static():
    path = "data/"
    df2 = pd.read_csv(path+"X_train.csv")
    df2.pop("word")
    df2.pop("word2")
    df2.pop("season")
    df2.pop("No")   
    
    df = df2[df2["label"] == 0]
    
    df.describe().to_csv("Line.csv")
    
    df = df2[df2["label"] == 1]
    
    df.describe().to_csv("Trans.csv")

    df = df2[df2["label"] == 2]
    
    df.describe().to_csv("Freq.csv")    
    
    # print(df2.describe())
    

  
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
  
def LF_1(df):  
    # df = train_x.copy()
    df["LF1"] = 0
    df.loc[(df.max_v_aup>0.1),"LF1" ] = 1
    return df
    
    # df3 = df1[(df1["max_v_dup"] > 0.01) | (df1["max_v_ddn"] >0.005)
    # | (df1["max_v_adn"] > 2)| (df1["max_v_aup"] > 0.07)]

def LF_2(df):  
    # df = train_x.copy()
    df["LF2"] = 0
    df.loc[(df.max_v_adn>2),"LF2" ] = 1
    return df

def LF_3(df):  
    # df = train_x.copy()
    df["LF3"] = 0
    df.loc[(df.max_v_dup>0.01)| (df["max_v_ddn"] >0.005)| (df.max_v_adn>2)| (df.max_v_aup>0.1),"LF3" ] = 1
    return df
    
def LF_4(df):  
    # df = train_x.copy()
    df["LF4"] = 0
    df.loc[(df["r1"] >3)| (df["r3"] >1),"LF4" ] = 2
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
            'RF', 
            # 'DT', 
            # 'SVM',

            # 'GBDT'
            ]  
            
            
        classifiers = {
        
                    # 'NB':naive_bayes_classifier,   
                      'KNN':knn_classifier,  
                       'LR':logistic_regression_classifier,  
                       'RF':random_forest_classifier,  
                       # 'DT':decision_tree_classifier,  
                      # 'SVM':svm_classifier,  
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
        X_train= LF_1(X_train)
        X_train= LF_2(X_train)
        X_train= LF_3(X_train)
        # X_train= LF_4(X_train)
        
        X_test= LF_1(X_test)
        X_test= LF_2(X_test)
        X_test= LF_3(X_test)
        # X_test= LF_4(X_test)
        
        tr["LF1"] = X_train["LF1"]
        tr["LF2"] = X_train["LF2"]
        tr["LF3"] = X_train["LF3"]
        # tr["LF4"] = X_train["LF4"]
        
        te["LF1"] = X_test["LF1"]
        te["LF2"] = X_test["LF2"]
        te["LF3"] = X_test["LF3"]
        # te["LF4"] = X_test["LF4"]
        tr.to_csv("data/wk_1_tr.csv",index = None)
        
        te.to_csv("data/wk_1_test.csv",index = None)
        # print("result save done")
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    

        
    
    
def main():
    s1 = timeit.default_timer()
    # fix_data()
    get_nosiy_label()
    # top()
    # div_half()
    # for  ii in range(1,20):
        # get_nosiy_label(ii)
    # fun()
    s2 = timeit.default_timer()
    print('Time:(min) ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

