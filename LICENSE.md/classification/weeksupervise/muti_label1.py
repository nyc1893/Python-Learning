"""
Use S1 dataset after removed planned and seperate PMU for training 3128
the labeling function 

and use S1+S2 data as testing result, save it to simulate without knowing the S2 information
"""

# from sklearn.datasets import make_moons
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

import pandas as pd
import numpy as np
import timeit
import time  
from sklearn import metrics  
import pickle as pickle  
import pandas as pd
 
def read_data():  
    path = '../pickleset/'

    p1 = open(path +"X_S1_rocof_6.pickle","rb")
    # pickle_in = open("X_train_6.pickle","rb")
    p2 = open(path+ "X_S2_rocof_6.pickle","rb")


    pk1 = pickle.load(p1)
    pk2 = pickle.load(p2)

    print('pk1.shape',pk1.shape)
    # print('pk2.shape',pk2.shape)
    # X_train=np.concatenate((pk1, pk2), axis=0)
    X_train = pk1
    X_test = pk2
    #X_train=X_train*X_train

    
    
    p1 = open(path + "y_S1_rocof_6.pickle","rb")
    p2 = open(path+ "y_S2_rocof_6.pickle","rb")


    pk1 = pickle.load(p1)
    pk2 = pickle.load(p2)
    print('pk1.shape',pk1.shape)
    # print('pk2.shape',pk2.shape)
    # y_train=np.concatenate((pk1, pk2), axis=0)
    y_train = pk1
    y_test = pk2
    

    #cropping input
    fps=60
    start_crop=int(fps*60*4)
    stop_crop=int(fps*60*8)

    X_train=X_train[:,:,start_crop:stop_crop,:]
    X_test = X_test[:,:,start_crop:stop_crop,:]
    # X_train, X_test, y_train, y_test=train_test_split(X_train, y_train, test_size=0.8)


    # print('X_train.shape',X_train.shape)
    # print('y_train.shape',y_train.shape)
    # print('X_test.shape',X_test.shape)
    # print('y_test.shape',y_test.shape)
    
    # print(X_train.shape )
    # print(y_train.shape )

    #number of classes
    num_classes=len(np.unique(y_train))

    num = 100

    #separate PMUs to make more events
    X_train, y_train=removePlanned(X_train, y_train)
    X_train, y_train= separatePMUs(X_train,y_train)
    X_train, y_train = shuffle(X_train, y_train)

    X_test, y_test=removePlanned(X_test, y_test)
    X_test, y_test= separatePMUs(X_test,y_test)
    X_test, y_test = shuffle(X_test,y_test)
    

    # X_train = X_train[0:num]
    # y_train = y_train [0:num]
    # X_test = X_test[0:num]
    # y_test = y_test[0:num]
    
    
    
    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)
    
    return X_train,y_train,X_test,y_test
    

        

def removePlanned(X,y):

    
    X_new=[]
    y_new=[]
    for i in range(len(y)):
        #print(i)
    
        if y[i]==0:
            y_new.append(0)
            X_new.append(X[i,:,:,:])
            # X_new.append(X[i,:::])
        elif y[i]==1:
            y_new.append(1)
            X_new.append(X[i,:,:,:])
            # X_new.append(X[i,:::])
            
        elif y[i]==2:
            y_new.append(2)
            # X_new.append(X[i,:::])
            X_new.append(X[i,:,:,:])

        elif y[i]==3:
            y_new.append(3)
            # X_new.append(X[i,:::])
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
    y_new=[]
    for i in range(len(y)):
        if y[i]==0:
            for j in range(num_pmu):
                y_new.append(0)
                
        if y[i]==1:
            for j in range(num_pmu):
                y_new.append(1)
                
        if y[i]==2:
            for j in range(num_pmu):
                y_new.append(2)
                
        if y[i]==3:
            for j in range(num_pmu):
                y_new.append(3)
        
    
    return X,np.array(y_new)    

    
  
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
  

  

      
if __name__ == '__main__':  
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  
    
    
    # train_x, train_y, test_x, test_y = read_data() 
    # print('train_x.shape',train_x.shape)
    # print('train_y.shape',train_y.shape)
    # print('test_x.shape',test_x.shape)
    # print('test_y.shape',test_y.shape)
    
    train_x, train_y, test_x, test_y = read_data()  
    X_train = train_x
    X_test = test_x
    
    y_test = test_y
    y_train = train_y
    
    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)
    X_train['label'] = y_train
    X_test = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test)
    X_test['label'] = y_test
    
    X_train.to_csv('X_train.csv',index =None)
    X_test.to_csv('X_test.csv',index =None)
    # X_train = train_x.values
    
    df = pd.concat([X_train,X_test])
    y_test = df.pop('label').values
    X_test = df.values
    
    # X_test = pd.concat([train_x,test_x]).values
    # y_test = pd.concat([train_y,test_y]).values
    
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
            'DT', 
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
        df.to_csv('label.csv',index =None)
        print('X_tain.shape',X_tain.shape)
        print('save done!')
        # if model_save_file != None:  
            # pickle.dump(model_save, open(model_save_file, 'wb')) 
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
