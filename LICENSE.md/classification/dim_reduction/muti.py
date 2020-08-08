

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import pandas as pd

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Reshape, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers,regularizers
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
# from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection  import StratifiedKFold
from sklearn.model_selection  import cross_val_score
from sklearn.model_selection import learning_curve, GridSearchCV  
from sklearn.svm import SVC    
from sklearn.decomposition import FastICA 
from sklearn import manifold 

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
import sys
import pickle
import timeit
import datetime

start = timeit.default_timer()


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






def read_data(l,num):  

    # num = 1
    path = '../S1+2'
    p2 = open(path+ "tr_set.pickle","rb")
    X_train, y_train= pickle.load(p2)


    p2 = open(path+ "va_set.pickle","rb")
    X_test, y_test= pickle.load(p2)
    y_train = y_train[:,l]
    
    y_test = y_test[:,l]    
    # print('X_train.shape',X_train.shape)
    # print('y_train.shape',y_train.shape)
    # print('X_test.shape',X_test.shape)
    # print('y_test.shape',y_test.shape)    


    # only shuffle X_train part 
    # X_train, y_train =removePlanned(X_train, y_train)
    X_train, y_train=separatePMUs(X_train, y_train)

    X_test, y_test= separatePMUs(X_test,y_test)
 
    df = pd.read_csv('../Ltest3_'+str(l)+'.csv')
    list = df.columns.values.tolist()
    
    y_test = df[list[num]]
    
    X_train = np.concatenate((X_train, X_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)
    

    p2 = open("../S3-va_set.pickle","rb")
    X_test,y_test = pickle.load(p2)
    y_test = y_test[:,l]
    
    X_test, y_test= separatePMUs(X_test,y_test)
    # print('X_train.shape',X_train.shape)
    # print('y_train.shape',y_train.shape)
    # print('X_test.shape',X_test.shape)
    # print('y_test.shape',y_test.shape)    
    print(list[num])
    
    return X_train,y_train,X_test,y_test,list[num]    
    
def fun(a,b):
    if a == b:
        return 1
    else:
        return 0
        
def check_X(X,y):
    j = 0 
    num = X.shape[0]
    list= []
    for i in range(0,num-1):
        d1 = X[i,0,:,:]
        # d11 = y[i] 
        d2 = X[i+1,0,:,:]
        # d22 = y[i+1] 
        c1 = X[i,7,:,:]
        c2 = X[i+1,7,:,:]
        
        
        # and  (c1 ==c2).all()
        if (d1 ==d2).all() and (c1 ==c2).all() and ((y[i] == 0 and y[i+1] ==1) or (y[i] == 1 and y[i+1] ==0) or  (y[i] == 1 and y[i+1] ==1)):
            list.append(i)
            list.append(i+1)

            j = j+1
    print('total:',j)
    return list
            
def get_label (X,y):
    # X,y = read_data3()
    ll = check_X(X,y)
    # print(ll)
    
    df = pd.DataFrame(y)
    df.columns = ['v']
    
    cc = np.zeros(y.shape[0], dtype=np.int)
    cc[ll] = 1
    df['LL'] = cc
    # print(df['LL'].value_counts())
    df['L1'] = df.apply(lambda x: fun(x.v,0),axis =1)
    
    # print(df['L1'].value_counts())
    # df['L1'] = df.apply(lambda x: fun(x.LL,1),axis =1)
    df.loc[ll,'L1'] = 1 
    # print(df['L1'].value_counts())
    
    df['L2'] = df.apply(lambda x: fun(x.v,1),axis =1)
    # print(df['L2'].value_counts())
    df.loc[ll,'L2'] = 1 
    # print(df['L2'].value_counts())
    
    df['L3'] = df.apply(lambda x: fun(x.v,2),axis =1)
    df['L4'] = df.apply(lambda x: fun(x.v,3),axis =1)
    df.pop('v')
    df.pop('LL')

    return df.values
    
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

    pk1,pk3=removePlanned(pk1,pk3)
    
    X_train = pk1[tr]
    y_train = pk3[tr]
    X_val = pk1[val]
    y_val = pk3[val]
    
    X_train,y_train = separatePMUs(X_train,y_train )
    X_val,y_val = separatePMUs(X_val,y_val)
    print(X_train.shape) 
    print(X_val.shape) 
    print(y_train.shape) 
    print(y_val.shape) 
    
    return X_train,y_train,X_val, y_val  

def check_y_test(y_test):
    # X_test,y_test = read_data()
    i = 0
    num = y_test.shape[0]/23
    # num = 2
    # j = 0
    flag = 0
    for i in range(1,int(num)+1):
        ind1 = 23*(i-1)
        ind2 = 23*i
        
        data =  y_test[ind1:ind2]
        # print(data)
        for j in range(1,23):
            if(data[0] != data[j]):
                flag = 1
        if flag == 1:
            print('Not equal in ', i)   
    return flag
  

best_accuracy = 0.0    
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer
# from factor_analyzer import FactorAnalyzer



def plot(k):

    X_train,y_train,X_test,y_test  = rd2(1)
    
    feat_cols = [ 'p'+str(i) for i in range(X_train.shape[1]) ]
    df = pd.DataFrame(X_test,columns=feat_cols)
    df['label'] = y_test
    
    FA = FactorAnalysis(n_components = k).fit_transform(df[feat_cols].values)
    # FA = FactorAnalyzer(n_factors = k,rotation = 'varimax').fit(df[feat_cols].values)
    
    # FA.fit(df[feat_cols].values)
    # ev,v = FA.get_eigenvalues()
    xval = range(FA.shape[0])
    
    # list= ['F0','F1','F2']
    plt.figure(figsize=(12,8))
    plt.title('Factor Analysis Components')
    for i in range(0,k):
        plt.scatter(xval, FA[:,i],label='F_'+str(i))


    plt.legend()
    

    plt.grid()
    plt.savefig('FA_'+str(k)) 
    
    


def clf_svm2(k):
    
    X_train,y_train,X_test, y_test = proc(k)

    
    
    name = 'FA' + str(k)
    fname = 'opt/svmp-'+str(name)
    model = SVC(kernel='rbf', probability=True)    
    param_grid = {'C': [1e-3,1e-2,1e-1,1, 1e2, 1e3, 1e4], 'gamma': [1e-3,1e-2,1e-4]}    
    grid_search = GridSearchCV(model, param_grid, n_jobs = -1, verbose=2)    
    grid_search.fit(X_train, y_train)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
        
        
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    model.fit(X_train, y_train)           
    #保存Model(注:save文件夹要预先建立，否则会报错)
    with open(fname+ '.pickle', 'wb') as f:
        pickle.dump(model, f)

    #读取Model
    with open(fname+'.pickle', 'rb') as f:
        clf2 = pickle.load(f)
        #测试读取后的Model

    

    train_out2 = clf2.predict(X_train)
    predict2 = clf2.predict(X_test)
    print(name)
    print('training accucy:',accuracy_score(train_out2,y_train))
    print('test accucy:',accuracy_score(predict2,y_test))            
    print(confusion_matrix(y_test,predict2))
    print(classification_report(y_test, predict2))  
    
def clf_svm():
    
    X_train,y_train,X_test, y_test = rd2(1)
    name = 'orginal'
    fname = 'opt/svmp-'+str(name)
    model = SVC(kernel='rbf', probability=True)    
    param_grid = {'C': [1e-3,1e-2,1e-1,1, 1e2, 1e3, 1e4], 'gamma': [1e-3,1e-2,1e-4]}    
    grid_search = GridSearchCV(model, param_grid, n_jobs = -1, verbose=2)    
    grid_search.fit(X_train, y_train)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
        
        
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    model.fit(X_train, y_train)           
    #保存Model(注:save文件夹要预先建立，否则会报错)
    with open(fname+ '.pickle', 'wb') as f:
        pickle.dump(model, f)

    #读取Model
    with open(fname+'.pickle', 'rb') as f:
        clf2 = pickle.load(f)
        #测试读取后的Model

    

    train_out2 = clf2.predict(X_train)
    predict2 = clf2.predict(X_test)
    print(name)
    print('training accucy:',accuracy_score(train_out2,y_train))
    print('test accucy:',accuracy_score(predict2,y_test))            
    print(confusion_matrix(y_test,predict2))
    print(classification_report(y_test, predict2))  

def main(k):
    # clf_svm()
    clf_svm2(k)
    
def IOSMAP(k):
    X_train,y_train,X_test, y_test = rd2(1)
    num  =  X_train.shape[0]     
    X_train=np.concatenate((X_train, X_test), axis=0)
    feat_cols = [ 'p'+str(i) for i in range(X_train.shape[1]) ]
    df = pd.DataFrame(X_train,columns=feat_cols)
    X = manifold.Isomap(n_neighbors=5, n_components=3, n_jobs=-1).fit_transform(df[feat_cols][:6000].values)
    
    X_train = X[:num,:]
    X_test = X[num:,:]
    print(X_train.shape) 
    print(X_test.shape) 

    return X_train,y_train,X_test,y_test    
    
def ICA(k):
 
    X_train,y_train,X_test, y_test = rd2(1)
    num  =  X_train.shape[0]
    
    X_train=np.concatenate((X_train, X_test), axis=0)
    ICA = FastICA(n_components=k, random_state=12) 
    
    feat_cols = [ 'p'+str(i) for i in range(X_train.shape[1]) ]
    df = pd.DataFrame(X_train,columns=feat_cols)
    X=ICA.fit_transform(df[feat_cols].values)
    

    X_train = X[:num,:]
    X_test = X[num:,:]
    print(X_train.shape) 
    print(X_test.shape) 

    return X_train,y_train,X_test,y_test
    
def FA(k):
 
    X_train,y_train,X_test, y_test = rd2(1)
    num  =  X_train.shape[0]
    
    X_train=np.concatenate((X_train, X_test), axis=0)
    
    feat_cols = [ 'p'+str(i) for i in range(X_train.shape[1]) ]
    df = pd.DataFrame(X_train,columns=feat_cols)

    FA = FactorAnalysis(n_components = k).fit_transform(df[feat_cols].values)   

    X_train = FA[:num,:]
    X_test = FA[num:,:]
    # print(X_train.shape) 
    # print(X_test.shape) 

    return X_train,y_train,X_test,y_test
    
if __name__ == '__main__':  
    k = int(sys.argv[1])
    s1 = timeit.default_timer()  
    # global best_accuracy 
    # for k in range(1,8):
    IOSMAP(k)
    # read_data(0)
    
    s2 = timeit.default_timer()  


    print ('Runing time is Hour:',round((s2 -s1)/3600,2))
