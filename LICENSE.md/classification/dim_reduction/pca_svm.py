

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

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

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
from get_label import cc,removePlanned,onlylabel,separatePMUs

start = timeit.default_timer()
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
def LF_fter(X_train,y_train,a1,b1):
    num = y_train.size

    X_new=[]
    y_new=[]
    

    for i in range(0,num):
        a = X_train[i]
        if (a.max()>a1 or a.min()<-b1):
            X_new.append(X_train[i])
            y_new.append(1)
        else :  
            X_new.append(X_train[i])
            y_new.append(0)

    return np.array(X_new), np.array(y_new)
  
def cc3(m):
    X_train,y_train = rd2(1)
    X_test,y_test = rd2(2)
    X_test, y_test= separatePMUs(X_test,y_test)
    X_train,y_train= separatePMUs(X_train,y_train)
    
    a1 = 0.1 
    b1 = 0.05
    X_train,y_train = LF_fter(X_train,y_train,a1,b1)
    X_test,y_test = LF_fter(X_test,y_test,a1,b1)    
    # df = csv_read
    df = pd.read_csv('snorkel4.csv')

    l = df.columns.values.tolist()
    name = l[m]
    y_test = df[name].values
    X_train = X_test
    y_train = y_test
    # X_train=np.concatenate((X_train, X_test), axis=0)
   
    # y_train=np.concatenate((y_train, y_test), axis=0)

    X_test,y_test = rd2(3)
    X_test, y_test= separatePMUs(X_test,y_test)
    X_test,y_test = LF_fter(X_test,y_test,a1,b1)    
    
    
    
    
    return X_train,y_train,X_test,y_test,name
    
def cc2(m):

    X_train,y_train,X_test,y_test,name = cc3(m)

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    temp = X_train
    X_train["max"] = temp.max(axis=1)#axis 0为列，1为行
    X_train["avg"] = temp.mean(axis=1)
    X_train["min"] = temp.min(axis=1)
    # X_train = X_train[['max','avg','min']]


    temp = X_test
    X_test["max"] = temp.max(axis=1)#axis 0为列，1为行
    X_test["avg"] = temp.mean(axis=1)
    X_test["min"] = temp.min(axis=1)
    # X_test = X_test[['max','avg','min']]
    
    # print(X_train.head())
    return X_train.values,y_train,X_test.values,y_test,name    
    


best_accuracy = 0.0    

def rd3(k):
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
    
    X_train,y_train =onlylabel(X_train,y_train)
    X_val, y_val  =onlylabel(X_val, y_val  )
    
    X_train,y_train =separatePMUs(X_train,y_train)
    X_val, y_val  =separatePMUs(X_val, y_val)    
    
    
    # print(X_train.shape) 
    # print(X_val.shape) 
    # print(y_train.shape) 
    # print(y_val.shape) 
    
    return X_train,y_train,X_val,y_val  
    
    

def read(v):

    X_train,y_train,X_test, y_test = cc()
    
    # X_train=np.concatenate((X_train, X_test), axis=0)
    X_train = X_test
    y_test,name = get_ll(v)
    # y_train=np.concatenate((y_train, y_test), axis=0)
    
    X_train = X_test
    y_train = y_test
    X_test, y_test, _,_= rd3(3)
    
    
    # print(X_train.shape) 
    # print(X_test.shape) 
    # print(y_train.shape) 
    # print(y_test.shape)    
    
    return X_train,y_train,X_test, y_test,name
    
def clf_svm(m):
    
    X_train,y_train,X_test, y_test,name = cc2(m)
    # name = 'filtered'
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
    # best_parameters = grid_search.best_estimator_.get_params()    
    # for para, val in list(best_parameters.items()):    
        # print(para, val)    
        
        



def std_PCA(**kwargs):
    scalar = MinMaxScaler()  # 用于数据预处理(归一化和缩放)
    pca = PCA(**kwargs)  # PCA本身不包含预处理
    pipline = Pipeline([('scalar', scalar), ('pca', pca)])
    return pipline

def pca_svm():
    X_train,y_train,X_test, y_test = cc() 
    v = 0
    ind = 0
    for k in range(100,120,2):
        # svd_solver='randomized'指定随机SVD,whiten=True做白化变换,让降维后的数据方差都为1
        model = std_PCA(n_components=k, svd_solver='randomized', whiten=True)
        # 注意,PCA的fit(即获得主成分特征矩阵U_reduce的过程)仅使用训练集!
        model.fit(X_train)
        # 训练集和测试集分别降维
        X_train_pca = model.transform(X_train)
        X_test_pca = model.transform(X_test)
        # 使用GridSerachCV选择最佳的svm.SVC模型参数
        param_grid = {'C': [1e-1,1e-2,1e1,1e2, 1e3], 'gamma': [1e-3,1e-2,1e-4] }
        # verbose控制verbosity,决定输出的过程信息的复杂度,=2详细
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, verbose=2, n_jobs=-1)
        # 这里用降维后的训练集来找最佳参数,则找到的最佳参数也会适合降维后的训练集
        clf = clf.fit(X_train_pca, y_train)
        print("best parameter: ", clf.best_params_)
        # 用找到的最佳参数对降维后的测试集做预测,可以直接用clf.best_estimator_获得训练好的最佳参数模型
        y_pred = clf.best_estimator_.predict(X_test_pca)
        acc = accuracy_score(y_test,y_pred)  
        
        if acc > v:
            v = acc
            ind = k
        
    print('k= ', ind)
    print('acc= ', v)
    # print(confusion_matrix(y_test,y_pred))
    # print(classification_report(y_test, y_pred))   
    
def pca_svm2(k):
    X_train,y_train,X_test, y_test = cc() 
    v = 0
    ind = 0
   
    # svd_solver='randomized'指定随机SVD,whiten=True做白化变换,让降维后的数据方差都为1
    model = std_PCA(n_components=k, svd_solver='randomized', whiten=True)
    # 注意,PCA的fit(即获得主成分特征矩阵U_reduce的过程)仅使用训练集!
    model.fit(X_train)
    # 训练集和测试集分别降维
    X_train_pca = model.transform(X_train)
    X_test_pca = model.transform(X_test)
    # 使用GridSerachCV选择最佳的svm.SVC模型参数
    param_grid = {'C': [1e-1,9e-2,8e-2,7e-2], 'gamma': [1e-3,9e-4,2e-3,3e-3] }
    # verbose控制verbosity,决定输出的过程信息的复杂度,=2详细
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, verbose=2, n_jobs=-1)
    # 这里用降维后的训练集来找最佳参数,则找到的最佳参数也会适合降维后的训练集
    clf = clf.fit(X_train_pca, y_train)
    print("best parameter: ", clf.best_params_)
    # 用找到的最佳参数对降维后的测试集做预测,可以直接用clf.best_estimator_获得训练好的最佳参数模型
    y_pred = clf.best_estimator_.predict(X_test_pca)
    acc = accuracy_score(y_test,y_pred)  

    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test, y_pred))   
    
        
def test_svm():
    
    X_train,y_train,X_test, y_test,name = read(0)  
    fname = 'opt/svmp-'+str(name)



    #读取Model
    with open(fname+'.pickle', 'rb') as f:
        clf2 = pickle.load(f)
        #测试读取后的Model

    

    train_out2 = clf2.predict(X_train)
    predict2 = clf2.predict(X_test)
    
    # print('training accucy:',accuracy_score(train_out2,y_train))
    # print('test accucy:',accuracy_score(predict2,y_test))            

    matrix=confusion_matrix(y_test,predict2)
    print(name)
    print(matrix)
    class_report=classification_report(y_test,predict2)
    print(class_report)

    
    
def main(k):    
    s1 = timeit.default_timer()  
    clf_svm(k)
    # test_svm()
    # pca_svm2(10)
    # cc2()
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
if __name__ == '__main__':  




    k = int(sys.argv[1])
    

    main(k)

    
