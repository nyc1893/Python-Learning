

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
import datapick
import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# python getlabel.py 2>&1 | tee b1.log
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
from pandas._testing import assert_frame_equal

# from rno_fun import data_lowfreq

    
    

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

    train_x, train_y,test_x, test_y =datapick.top()   

    
 
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

             # 'KNN', 
             # 'LR', 
            'RF', 
            # 'DT', 
            # 'SVM',

            # 'GBDT'
            ]  
            
            
        classifiers = {
        
                    # 'NB':naive_bayes_classifier,   
                      # 'KNN':knn_classifier,  
                       # 'LR':logistic_regression_classifier,  
                       'RF':random_forest_classifier,  
                       # 'DT':decision_tree_classifier,  
                      # 'SVM':svm_classifier,  
                    # 'SVMCV':svm_cross_validation,  
                     # 'GBDT':gradient_boosting_classifier  
        }  
          
        print('reading training and testing data...')  
        # 
        print('train_y.shape',y_test.shape)
        tr = pd.DataFrame(y_train)
        te = pd.DataFrame(y_test)
        
        df  = y_test[:,np.newaxis]
        df2 = y_train[:,np.newaxis]
        te.columns = ['real']
        list = []
        list.append('real')
        

        for classifier in test_classifiers:  
            print('******************* %s ********************' % classifier)  
            start_time = time.time()  
            model = classifiers[classifier](train_x, train_y)  
            print('training took %fs!' % (time.time() - start_time))  
            predict = model.predict(X_test) 
            train_out = model.predict(X_train)  
            
            tr[classifier] = train_out
            te[classifier] = predict
            # change(predict,cc)
            print('training error')  
            # print(type(train_out))
            # print(type(y_train))
            matrix=confusion_matrix(train_out, y_train)
            print(matrix)
            class_report=classification_report(train_out, y_train)
            print(class_report)

            print('testing error') 
            matrix=confusion_matrix(y_test, predict)
            print(matrix)
            class_report=classification_report(y_test, predict)
            print(class_report)

        # tr.to_csv("ml/tr.csv",index = None)
        te.to_csv("te.csv",index = None)
        # print("result save done")
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
# def change(pred,cc):
    # print(pred.shape)
    # print(cc.shape)
    # for i in range(0,pred.shape[0]):
        # if(cc[i] == 1 and pred[i]==0):
            # pred[i] = 1
    # return pred


    


def get_split(ii):        
    path = "ml/"
    # df2 = pd.read_csv(path+"savefreq_10m_23_"+str(ii)+".csv")
    df = pd.read_csv(path+"ss"+str(ii)+".csv")
    df2 = df.replace([np.inf, -np.inf], np.nan)
    df2 = df2.dropna()
    y = df2["label"].values    
    
    df.to_csv(path+"ss"+str(ii)+".csv",index =None)

    spliter(ii,y)

def spliter(num,y3):        
    a = np.arange(0,y3.shape[0])
    tr,val = train_test_split(a,test_size=0.2)   
    print(tr.shape)
    print(val.shape)
    path2 = 'index5/'
    np.save(path2+'tr_'+str(num)+'.npy',tr) 
    np.save(path2+'val_'+str(num)+'.npy',val)

def mid_data(ii):

    path2 = 'index5/'
    tr = np.load(path2+'tr_'+str(ii)+'.npy') 
    val = np.load(path2+'val_'+str(ii)+'.npy') 
    list1 = tr.astype(int).tolist()
    list2 = val.astype(int).tolist()
    path = "ml/"

    df2 = pd.read_csv(path+"ss"+str(ii)+".csv")
    
    df2 = df2.replace([np.inf, -np.inf], np.nan)
    df2 = df2.dropna()
    y = df2.pop("label")

    X = df2
    
    X_train = X.iloc[list1]
    y_train = y.iloc[list1]
    X_val = X.iloc[list2]
    y_val = y.iloc[list2]

    return X_train,y_train,X_val,y_val
    

def top():
    X_train,y_train,X_val,y_val = mid_data(1)
    for ii in range(2,14):
        X_train2,y_train2,X_val2,y_val2 = mid_data(ii)
        X_train = pd.concat([X_train, X_train2], axis=0)
        y_train = pd.concat([y_train, y_train2], axis=0)
        
        X_val = pd.concat([X_val, X_val2], axis=0)
        y_val = pd.concat([y_val, y_val2], axis=0)       
    
    sav = 0
    if sav == 1:
        
        X_train['label'] = y_train
        X_val['label'] = y_val
        X_train.to_csv("ml/X_train.csv",index =None)
        X_val.to_csv("ml/X_val.csv",index =None)
    # X_train.pop("S")
    # X_val.pop("S")
    # X_train.pop("No")
    # X_val.pop("No")   
    # X_train.pop("eve_name")
    # X_val.pop("eve_name")      
    # X_train.pop("flag")
    # cc = X_val.pop("flag").values
    
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)

    # print(X_train.head())
    # print(y_train.head())
    # print(np.where(y_train == 1)[0])
    # print(np.where(y_val == 1)[0])
    return X_train,y_train,X_val,y_val
    
    
    


    
def main():
    s1 = timeit.default_timer()  
    # for i in range(1,14):
        # get_split(i)
        
    # mid_data(1)
    run()
    # cc(12)
    # top()
    # pack_all()
    s2 = timeit.default_timer()
    #running time
    print('Time: ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

