

# import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import math
import pandas as pd
import time  
from sklearn import metrics  
import pickle as pickle  
import pandas as pd
# refer to:
# https://www.datacareer.ch/blog/parameter-tuning-in-gradient-boosting-gbm-with-python/
import pickle
import datapick2
# import datapick3
import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# python getlabel2.py 2>&1 | tee b2.log
# cat /proc/cpuinfo 2>&1 | tee b2.log


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split



import timeit
import time  
from sklearn import metrics  
import pickle as pickle  
import pandas as pd
# from pandas._testing import assert_frame_equal

# from rno_fun import data_lowfreq

    
    
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

  

      
def base():
    # data_file = "H:\\Research\\data\\trainCG.csv"  
    s1 = timeit.default_timer()  

    X_train, y_train,X_test, y_test =datapick2.top()   


    
    baseline = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1500,max_depth=7, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10)
    baseline.fit(X_train, y_train)
    predictors=list(X_train)
    plt.figure( figsize=(12,6))
    feat_imp = pd.Series(baseline.feature_importances_, predictors).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Importance of Features')
    plt.ylabel('Feature Importance Score')
    plt.tight_layout()
    plt.savefig('cc')
    print('Accuracy of the GBM on test set: {:.3f}'.format(baseline.score(X_test, y_test)))
    pred=baseline.predict(X_test)
    print(classification_report(y_test, pred))


def tt():
    # p_test3 = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,200,300,650,1000,1500]}
    X_train, y_train,X_test, y_test =datapick2.top()  
    p_test4 = {'min_samples_split':[2,4,6,8,10,20,40,60,100], 'min_samples_leaf':[1,3,5,7,9]}

    tuning = GridSearchCV(estimator =GradientBoostingClassifier(learning_rate=0.01, n_estimators=1500,max_depth=7, subsample=1,max_features='sqrt', random_state=10), 
                param_grid = p_test4, scoring='accuracy',n_jobs=4,iid=False, cv=5)
    tuning.fit(X_train,y_train)
    print(tuning.best_params_)


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
    

def rd_rof(ii):
    path1 = "data/"
    df  = pd.read_csv(path1 +'Ss'+str(ii)+'.csv')
    full_ind = df.index
    path2 = "../../../../../"
    dg = pd.read_csv(path2 +'muti.csv')
    # print("dg.shape",dg.shape)
    dg = dg[(dg["v"]==7)]
    ll = dg["new"].values.tolist()
    print(df.shape[0])
    st = []
    
    dt2 = df.loc[df["word"].str.contains("Line_Trip|Line_Lightning")==True ].index
    dt4 = df.loc[df["word"].str.contains("Transformer")==True ].index
    dt3 = df.loc[df["word"].str.contains("Transformer_Trip|Transformer_Lightning|Transformer_Planned")].index
    dt1 = df.loc[df.word2.isin(ll)].index
    ind1 = bingji(dt2,dt4)
    ind1 = bingji(dt1,ind1)
    # print(len(dt2))

    # print("  "+str(len(dt1)))
    # print(len(ind1))
    #step1 get rid of the data
    ind3 = list(set(full_ind)^set(ind1))
    ind2 = bingji(dt3,ind3)
    # print(len(ind2))


    #step2 label data again 
    
    dg = pd.read_csv(path2 +'muti.csv')
    # print("dg.shape",dg.shape)
    dg = dg[(dg["v"]==4)]
    ll = dg["new"].values.tolist()
    dt1 = df.loc[df.word2.isin(ll)].index
    dt2 = df.loc[df["word"].str.contains("Trans")==True ].index
    
    
    dg = pd.read_csv(path2 +'muti.csv')
    # print("dg.shape",dg.shape)
    dg = dg[(dg["v"]==5)]
    ll = dg["new"].values.tolist()
    dt3 = df.loc[df.word2.isin(ll)].index
    dt4 = df.loc[df["word"].str.contains("Freq")==True ].index    
    
    
    df.label = 0

    df.label[dt2] = 1
    df.label[dt1] = 1
    
    
    df.label[dt3] = 2
    df.label[dt4] = 2
    
    
    res = df.iloc[ind2]
    res.pop("word")
    res.pop("word2")
    # res.pop("season")
    # res.pop("No")
    
    
    
    res["r1"] = res["mean_r_aup"]/res["mean_r_adn"]
    res["r2"] = res["mean_r_aup"]/(res["mean_r_adn"]+res["mean_r_aup"])
    res["r3"] = res["mean_r_up"]/res["mean_r_dn"]
    
    # res["r4"] = res["mean_v_aup"]/res["mean_v_adn"]
    # res["r5"] = res["mean_v_aup"]/(res["mean_v_adn"]+res["mean_v_aup"])
    
    # res["r6"] = res["mean_i_aup"]/res["mean_i_adn"]
    # res["r7"] = res["mean_i_aup"]/(res["mean_i_adn"]+res["mean_i_aup"])
    
    # res["r8"] = res["max_v_dn"]/res["mean_v_dn"]
    # res["r9"] = res["max_i_dn"]/res["mean_i_dn"]
    
    # res["r10"] = res["max_v_up"]/res["mean_v_up"]
    # res["r11"] = res["max_i_up"]/res["mean_i_up"]
    
    y = res.pop("label")
    res = res.iloc[:,]
    res = res.replace([np.inf, -np.inf], np.nan)
    ind = res[res.isnull().T.any()].index
    res.loc[ind,:] = -1
    return res,y

def top():
    X_train,y_train,X_val,y_val = datapick3.mid_data(1)
    for ii in range(2,14):
        X_train2,y_train2,X_val2,y_val2 = datapick3.mid_data(ii)
        X_train = pd.concat([X_train, X_train2], axis=0)
        y_train = pd.concat([y_train, y_train2], axis=0)
        
        X_val = pd.concat([X_val, X_val2], axis=0)
        y_val = pd.concat([y_val, y_val2], axis=0)       
    
    sav = 1
    if sav == 1:
        
        X_train['label'] = y_train
        X_val['label'] = y_val
        X_train.to_csv("data/X_train.csv",index =None)
        X_val.to_csv("data/X_val.csv",index =None)
        
    # else:
        # X_train.pop("season")
        # X_val.pop("season")
        # X_train.pop("No")
        # X_val.pop("No")   

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
    
    
    
def find_err():
    path = "data/"
    df3 = pd.read_csv("te3.csv")
    # df3 = pd.read_csv(path+"X_val.csv")
    # df3 = df3[["season","No"]]
    # df3 = pd.concat([df3,df2],axis =1)
    ind= df3[df3["real"]!=df3["RF"]].index
    df3 = df3.iloc[ind] 
    print(df3.head())
    print(ind)
    df3.to_csv(path+"err.csv",index =None)
    
def jiaoji(listA,listB):
    return list(set(listA).intersection(set(listB)))

    
    # let real value change to the muti- labels
def rev_res():
    df1 = pd.read_csv("data/X_val.csv")
    df2 = pd.read_csv("te2.csv")
    df2["word2"] = df1["word2"]
    df2["season"] = df1["season"]
    df2["No"] = df1["No"]
    path = "../../../../../"
    dg = pd.read_csv(path +"muti.csv")
    
    dg = dg[(dg["v"]==4)]
    ll = dg["new"].values.tolist()    
    dt1 = df2.loc[df2.word2.isin(ll)].index
    dt2 = df2[df2["real"]!=df2["GBDT"]].index
    dt3 = jiaoji(dt1,dt2)
    
    df2.loc[dt3,"real"] = df2.loc[dt3,"GBDT"] 
    
    df2.to_csv("te3.csv",index = None)

def replot():
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    df = pd.read_csv("te3.csv")
    df.pop("No")
    df.pop("word2")
    df.pop("season")
    y_test = df["real"]
    
    # ll = df.columns.values
    ll =["RF","GBDT"]
    for i in ll:
        
        y_pred = df[i]
        print('\n testing error of ' + str(i)) 
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
    # for i in range(1,14):
        # get_split(i)
    # top()
    
    # mid_data(1)
    # run()
    # base()
    tt()
    # rev_res()
    # find_err()
    # estimate()
    # replot()

    s2 = timeit.default_timer()
    #running time
    print('Time: ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

