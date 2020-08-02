

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
start = timeit.default_timer()

def onlylabel(X,y):
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
    
        # elif y[i]==1:
            # y_new.append(1)
            # X_new.append(X[i,:,:,:])
    
            
        elif y[i]==2:
            y_new.append(1)
            X_new.append(X[i,:,:,:])
        
        # elif y[i]==3:
            # y_new.append(3)
            # X_new.append(X[i,:,:,:])
        

    return  np.array(X_new), np.array(y_new)

def rd2(k):
    path1 = '../../pickleset/'

    list = ['rocof','v_grad','i_grad', 'vp_a_diff_grad', 'ip_a_diff_grad','f_grad']
    
    p1 = open(path1 +'X_S'+str(k)+'_'+str(list[0])+'_6.pickle',"rb")
    pk1 = pickle.load(p1)
    # len(list)
    # for i in range(1,len(list)):
        # p2 = open(path1 +'X_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")
        # pk2 = pickle.load(p2)    
    
        # pk1=np.concatenate((pk1, pk2), axis=3)
        
    fps=60
    start_crop=int(fps*60*4)
    stop_crop=int(fps*60*8)

    pk1=pk1[:,:,start_crop:stop_crop,:]
    
    p3 = open(path1 + 'y_S'+str(k)+'_rocof_6.pickle',"rb")
    pk3 = pickle.load(p3)        
    
    path2 = '../../cnn2/index/'
    tr=np.load(path2 +'tr_' +str(k)+'.npy')
    val=np.load(path2 +'val_' +str(k)+'.npy')
    tr=tr.tolist()  
    val=val.tolist() 

    pk1,pk3=onlylabel(pk1,pk3)
    
    X_train = pk1
    y_train = pk3

    
    print(X_train.shape) 
    # print(X_val.shape) 
    print(y_train.shape) 
    # print(y_val.shape) 
    
    return X_train,  y_train
    
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
  

  
def cc():
    X_train,y_train = rd2(1)
    X_test,y_test = rd2(2)
    X_test, y_test= separatePMUs(X_test,y_test)
    X_train,y_train= separatePMUs(X_train,y_train)
    return X_train,y_train,X_test,y_test
      
def main(l):
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  

    train_x, train_y, test_x, test_y = cc() 
    # print('train_x.shape',train_x.shape)
    # print('train_y.shape',train_y.shape)
    # print('test_x.shape',test_x.shape)
    # print('test_y.shape',test_y.shape)
    

    X_train = train_x
    X_test = test_x
    
    y_test = test_y
    y_train = train_y
 
    
    print(X_test.shape)
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
          
        print('reading training and testing data...')  
        # 
        print('train_y.shape',y_test.shape)
        df  = y_test[:,np.newaxis]
        list = []
        list.append('real')
        for classifier in test_classifiers:  
            print('******************* %s ********************' % classifier)  
            start_time = time.time()  
            model = classifiers[classifier](train_x, train_y)  
            print('training took %fs!' % (time.time() - start_time))  
            predict = model.predict(X_test)  
            # np.save(classifier+"_pred.npy",predict)
            print(predict.shape)
            predict = predict[:,np.newaxis]
    
            df = np.concatenate((df,predict),axis=1)
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
        df.to_csv('sn_label'+str(l)+'.csv',index =None)
        print('X_tain.shape',X_tain.shape)
        print('save done!')
        # if model_save_file != None:  
            # pickle.dump(model_save, open(model_save_file, 'wb')) 
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))


    
    
if __name__ == '__main__':  

    main(0)

    # read_data(0)
