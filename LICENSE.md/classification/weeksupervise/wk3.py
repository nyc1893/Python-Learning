# from sklearn.datasets import make_moons
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle as pickle  
import pandas as pd
import numpy as np
import getlabel2
import timeit
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
# python wk3.py 2>&1 | tee sno_test.log

import wk1_test
def top(ii):
    df1 = pd.read_csv("data/wk_1_tr.csv")
    df2 = pd.read_csv("data/wk_1_test.csv")
    df3 = pd.read_csv("wk_2_test.csv")
    # print(df1.shape)
    # print(df2.shape)
    # print(df3.shape)
    # print(df1.shape[0] +df3.shape[0])
    X_train = pd.read_csv("data/x1.csv")
    X_test = pd.read_csv("data/x2.csv")
    # print(X_train.shape)
    # print(X_test.shape)    
    
    
    X_train.pop("label") 
    
    
    ll = ["Maj","SNO","rf_pred"]
    st1 = df1["real"].values.tolist()
    st2 = df3[ll[ii]].values.tolist()

    
    
    
    tr_label = st1+st2
    tr_label = np.array(tr_label)
    X_test.pop("season")
    X_test.pop("No")
    X_test.pop("word")
    X_test.pop("word2")  
    
    X_train.pop("season")
    X_train.pop("No")
    X_train.pop("word")
    X_train.pop("word2")  
    
    y_test  = X_test.pop("label").values
    print(X_train.shape)
    print(tr_label.shape)
    print(X_test.shape)
    print(y_test.shape)    
    return X_train,tr_label,X_test,y_test
    
def top2(ii):
    df1 = pd.read_csv("data/wk_1_tr.csv")
    df2 = pd.read_csv("data/wk_1_test.csv")
    df3 = pd.read_csv("wk_2_test.csv")
    # print(df1.shape)
    # print(df2.shape)
    # print(df3.shape)
    # print(df1.shape[0] +df3.shape[0])
    X_train = pd.read_csv("data/x1.csv")
    X_test = pd.read_csv("data/x2.csv")
    # print(X_train.shape)
    # print(X_test.shape)    
    
    a = int(0.33*X_train.shape[0])
    X_train = X_train.iloc[a:]
    X_train.pop("label") 
    
    
    ll = ["Maj","SNO","rf_pred"]
    st1 = df1["real"].values.tolist()
    st2 = df3[ll[ii]]


    tr_label = st2
    X_test.pop("season")
    X_test.pop("No")
    X_test.pop("word")
    X_test.pop("word2")  
    
    X_train.pop("season")
    X_train.pop("No")
    X_train.pop("word")
    X_train.pop("word2")  
    
    y_test  = X_test.pop("label").values
    print(X_train.shape)
    print(tr_label.shape)
    print(X_test.shape)
    print(y_test.shape)    
    return X_train,tr_label,X_test,y_test
# and could change sno to other model...

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
        # print("result save done")
    # s2 = timeit.default_timer()  
    # print ('Runing time is (mins):',round((s2 -s1)/60,2))
def jiaoji(listA,listB):
    return list(set(listA).intersection(set(listB)))

def rev_res():
    # df1 = pd.read_csv("data/X_val.csv")
    X_train,df1 = wk1_test.filter_data()
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
    df.pop("No")
    df.pop("word2")
    df.pop("season")
    y_test = df["real"]
    
    ll = df.columns.values
    print(type(ll))
    # for i in range(1,ll.shape[0]):
        
    y_pred = df["RF"]
    print('\n testing error of RF') 
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

    print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))    
    
        
        
    matrix=confusion_matrix(y_test, y_pred)
    print(matrix)
    class_report=classification_report(y_test, y_pred)
    print(class_report)

def main():
    s1 = timeit.default_timer()  
    # div_half()
    # top2(1)
    for  ii in range(0,3):
        get_test_label(ii)
        rev_res()
        replot()
    s2 = timeit.default_timer()
    print('Time:(min) ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

