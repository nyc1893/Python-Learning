# Find the right event classification classification model

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle as pickle  
import pandas as pd
import numpy as np
import getlabel2
import timeit
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
# python wk3.py 2>&1 | tee sno_test.log

import wk1

# only truth label 
def top1():

    df3 = pd.read_csv("wk_2_test.csv")


    df2 = open("data/temp_data.pickle","rb")
    df2 = pickle.load(df2)
    
    X_train,y1,X_val,y2 = df2


    X_test = pd.read_csv("data/x2.csv")

    # path = "../../../../../"
    # dg = pd.read_csv(path +"muti.csv")
    # dg = dg[(dg["v"]==4)]
    # ll = dg["new"].values.tolist()    
    # dt1 = df1.loc[df1.word2.isin(ll)].index
    # dt2= df1.index 

    # dt3 = list(set(dt2) - set(dt1))
    # X_test = df1.iloc[dt3]



    st1 = y1.values
    
    tr_label =  st1

    X_test.pop("season")
    X_test.pop("No")
    X_test.pop("word")
    X_test.pop("word2")  
    
    X_train.pop("season")
    X_train.pop("No")
    X_train.pop("word")
    X_train.pop("word2")  
    

    y_test  = X_test.pop("label").values

    return X_train,(tr_label),X_test,(y_test)
    
# truth label + estimated label
def top2(ii):

    df3 = pd.read_csv("wk_2_test.csv")
    # ind1 = df3[df3["SNO"]!=0].index

    df2 = open("data/temp_data.pickle","rb")
    df2 = pickle.load(df2)
    
    X_train,y1,X_val,y2 = df2

    X_train = pd.concat([X_train,X_val],axis=0)
    # print(X_train.head())
    # X_train = pd.read_csv("data/x1.csv")
    X_test = pd.read_csv("data/x2.csv")

    # path = "../../../../../"
    # dg = pd.read_csv(path +"muti.csv")
    # dg = dg[(dg["v"]==4)]
    # ll = dg["new"].values.tolist()    
    # dt1 = df1.loc[df1.word2.isin(ll)].index
    # dt2= df1.index 

    # dt3 = list(set(dt2) - set(dt1))
    # X_test = df1.iloc[dt3]
    
    
    # dt1 = X_test[X_test["label"]==0].index
    # dt2 = X_test[X_test["label"]==1].index    
    # dt3 = X_test[X_test["label"]==2].index

    # dff1 = X_test.iloc[dt1[:190]]
    # dff2 = X_test.iloc[dt2]
    # dff3 = X_test.iloc[dt3]    
    
    # X_test = dff1
    # X_test = pd.concat([X_test,dff2],axis=0)
    # X_test = pd.concat([X_test,dff3],axis=0)
    


    # ll = ["SNO","SelfTrain","Maj","rf_pred"]
    
    
    st1 = y1.values
    st2 = df3[lp[ii]].values
    
    tr_label =  np.concatenate((st1,st2))
    # print (st1.shape)
    # print (st2.shape)
    # print (tr_label.shape)
    # print (st1[:5])
    # print (st2[:5])    
    
    
    # tt = st1+st2
    # tt = np.array(tt)
    # print(tt[:50])
    X_test.pop("season")
    X_test.pop("No")
    X_test.pop("word")
    X_test.pop("word2")  
    
    X_train.pop("season")
    X_train.pop("No")
    X_train.pop("word")
    X_train.pop("word2")  
    
    # X_train["SNO"] = tr_label
    # X_train = X_train[X_train["SNO"]!=-1]
    # tr_label = X_train.pop("SNO").values
    y_test  = X_test.pop("label").values
    # print(X_train.shape)
    # print(tr_label.shape)
    # print(X_test.shape)
    # print(y_test.shape)    
    # return X_train,detrans(tr_label),X_test,detrans(y_test)
    return X_train,(tr_label),X_test,(y_test)
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
    # model = GradientBoostingClassifier(n_estimators=200)  
    model = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1500,max_depth=7, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10) 
    model.fit(train_x, train_y)  
    return model  
  
  
# SVM Classifier  
def svm_classifier(train_x, train_y):  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    model.fit(train_x, train_y)  
    return model  
  
def svm_cross_validation(train_x, train_y): 
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC 
    model = SVC(kernel='rbf', probability=True) 
    # param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]} 
    param_grid = {'C': [ 1e-1, 1, 10], 'gamma': [0.001, 0.0001]} 
    grid_search = GridSearchCV(model, param_grid, n_jobs = 39, verbose=1) 
    grid_search.fit(train_x, train_y) 
    best_parameters = grid_search.best_estimator_.get_params() 
    for para, val in list(best_parameters.items()): 
        print(para, val) 
        model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True) 
        model.fit(train_x, train_y) 
    return model 
  
  
def get_test_label(ii):
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    # s1 = timeit.default_timer()  

    train_x, train_y,test_x, test_y = top2(ii)   

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
            'DT', 
            'SVM',
            # 'SVMCV',
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
            # print('******************* %s ********************' % classifier)  
            # start_time = time.time()  
            model = classifiers[classifier](X_train, y_train)  
            # print('training took %fs!' % (time.time() - start_time))  
            predict = model.predict(X_test) 
            train_out = model.predict(X_train)  
            
            tr[classifier] = train_out
            te[classifier] = predict


        # tr.to_csv("wk_1_tr.csv",index = None)
        
        te.to_csv("te2.csv",index = None)



def get_normal_label():
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    # s1 = timeit.default_timer()  

    train_x, train_y,test_x, test_y = top1()   

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
            'DT', 
            # 'SVM',
            'SVMCV'
            'GBDT'
            ]  
            
            
        classifiers = {
        
                    # 'NB':naive_bayes_classifier,   
                      'KNN':knn_classifier,  
                       'LR':logistic_regression_classifier,  
                       'RF':random_forest_classifier,  
                       'DT':decision_tree_classifier,  
                      # 'SVM':svm_classifier,  
                    'SVMCV':svm_cross_validation,  
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
            # print('******************* %s ********************' % classifier)  
            # start_time = time.time()  
            model = classifiers[classifier](X_train, y_train)  
            # print('training took %fs!' % (time.time() - start_time))  
            predict = model.predict(X_test) 
            train_out = model.predict(X_train)  
            
            tr[classifier] = train_out
            te[classifier] = predict


        # tr.to_csv("wk_1_tr.csv",index = None)
        
        te.to_csv("te2.csv",index = None)


def jiaoji(listA,listB):
    return list(set(listA).intersection(set(listB)))

def rev_res():
    # df1 = pd.read_csv("data/X_val.csv")
    X_train,df1 = wk1.filter_data()
    df2 = pd.read_csv("te2.csv")
    df2["word2"] = df1["word2"]
    df2["season"] = df1["season"]
    df2["No"] = df1["No"]
    path = "../../../../../"
    dg = pd.read_csv(path +"muti.csv")
    
    dg = dg[(dg["v"]==4)]
    ll = dg["new"].values.tolist()    
    dt1 = df2.loc[df2.word2.isin(ll)].index
    dt2 = df2[df2["real"]!=df2["RF"]].index
    dt3 = jiaoji(dt1,dt2)
    
    df2.loc[dt3,"real"] = df2.loc[dt3,"RF"] 
    
    df2.to_csv("te3.csv",index = None)

def replot():
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    df = pd.read_csv("te3.csv")
    # df.pop("No")
    # df.pop("word2")
    # df.pop("season")
    y_test = df["real"]
    
    ll = df.columns.values
    # print(type(ll))
    # for i in range(1,ll.shape[0]):
    res =[]
    y_pred = df["RF"]
    print('\n testing error of RF') 
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))
    print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))
    temp = round(accuracy_score(y_test, y_pred), 3)  
    res.append(temp)
    temp = round(precision_score(y_test, y_pred, average='macro'), 3)  
    res.append(temp)
    
    temp = round(recall_score(y_test, y_pred, average='macro'), 3)  
    res.append(temp)
    temp = round(f1_score(y_test, y_pred, average='macro'), 3)  
    res.append(temp)
    
    # if(ii== "SNO"):
    # matrix=confusion_matrix(y_test, y_pred)
    # print(matrix)
    # class_report=classification_report(y_test, y_pred)
    # print(class_report)
    

    # res.append(recall_score(y_test, y_pred, average='macro'))
    # res.append(f1_score(y_test, y_pred, average='macro'))
        
    # res = np.array(res)
    return res


def rev_res2(j):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    # df1 = pd.read_csv("data/X_val.csv")
    X_train,df1 = wk1.filter_data()
    df2 = pd.read_csv("te2.csv")
    df2["word2"] = df1["word2"]
    df2["season"] = df1["season"]
    df2["No"] = df1["No"]
    path = "../../../../../"
    dg = pd.read_csv(path +"muti.csv")
    
    dg = dg[(dg["v"]==4)]
    ll = dg["new"].values.tolist()    
    dt1 = df2.loc[df2.word2.isin(ll)].index
    
    # res = []

    st =[]
    dt2 = df2[df2["real"]!=df2[j]].index
    dt3 = jiaoji(dt1,dt2)
    df2.loc[dt3,"real"] = df2.loc[dt3,j] 
    y_pred = df2[j]
    y_test = df2["real"]
    # print('\n testing error of '+ j) 
    temp = round(accuracy_score(y_test, y_pred), 3)  
    st.append(temp)
    temp = round(precision_score(y_test, y_pred, average='macro'), 3)  
    st.append(temp)
    
    temp = round(recall_score(y_test, y_pred, average='macro'), 3)  
    st.append(temp)
    temp = round(f1_score(y_test, y_pred, average='macro'), 3)  
    st.append(temp)
    # res.append(st)
    return st

lp = ["SNO"]
com_list =["RF","KNN","LR","DT","SVM","GBDT"]
# lp = ["SNO","semi-DT","semi-GBDT","semi-SVC","semi-LR","semi-KNN","Maj"]
def run():
    # s1 = timeit.default_timer()  
    # div_half()
    # top2(0)
    # top1(0)
    # ll = ["SNO","rf_pred","SelfTrain","LabelSpread","Maj","rf_pred"]
    

    # len(lp)
    res =[]
    # for ii in range(len(com_list)):
    get_test_label(0)
    for ii in com_list:
        
        # rev_res()
        # print('\n testing error of '+ll[ii]) 
        # res.append(replot2())
        res.append(rev_res2(ii))
    # res = np.array(res)

    return res
    
    # get_normal_label()
    # rev_res()
    # print('\n testing error of limited') 
    # replot()        
        
    # s2 = timeit.default_timer()
    # print('Time:(min) ', (s2 - s1)/60 )
    
def main():
    # run()
    top2(0)

if __name__ == '__main__':  

    main()

