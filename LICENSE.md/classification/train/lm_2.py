"""
Load the best result of BO_DNN2.py
"""
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
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers,regularizers

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

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




    
def read_data():  
    path2 = '../svm/'
    df1 = pd.read_csv(path2 +'X_train.csv')
    df2 = pd.read_csv(path2 +'X_test.csv')
    df1 = pd.concat([df1,df2])
    
    path = '../pickleset/'
    p2 = open(path+ "X_S2_rocof_6.pickle","rb")
    pk2 = pickle.load(p2)
    X_test = pk2
    p2 = open(path+ "y_S2_rocof_6.pickle","rb")
    pk2 = pickle.load(p2)
    y_test = pk2
    
    y_train = df1.pop('label')
    X_train = df1
    
    
    fps=60
    start_crop=int(fps*60*4)
    stop_crop=int(fps*60*8)

    X_test= X_test[:,:,start_crop:stop_crop,:]
    X_test, y_test=removePlanned(X_test, y_test)
    X_test, y_test= separatePMUs(X_test,y_test)
    # X_test, y_test = shuffle(X_test,y_test)    
    
    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)    
    
    # X_test = X_test[0:23*10]
    # y_test = y_test[0:23*10]
    return X_train,y_train,X_test,y_test    
    
best_accuracy = 0.0    

def main():

    _,_,X_test,y_test = read_data()
    
    string = 'bodnn-best_model_so_far'
    model = load_model(str(string)+'.h5')  
    # y_pred_percentage=model.predict(X_val)
    y_pred=model.predict_classes(X_test) 
    matrix=confusion_matrix(y_test, y_pred)
    print(matrix)
    # print(name+'-'+str(i)+string)    
    class_report=classification_report(y_test, y_pred)
    print(class_report) 
    
    
if __name__ == '__main__':  
    # global best_accuracy 
    main()   
    
