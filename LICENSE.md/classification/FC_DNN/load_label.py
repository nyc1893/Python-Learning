#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:32:48 2020

@author: imanniazazari
"""


import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
import math
import sys
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

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
from sklearn.utils import shuffle
import pickle


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


def read_data(l):  

    # num = 1
    path = '2016-'
    p2 = open(path+ "tr_set.pickle","rb")
    X_train, y_train= pickle.load(p2)


    p2 = open(path+ "va_set.pickle","rb")
    X_test, y_test= pickle.load(p2)
    y_train = y_train[:,l]
    
    y_test = y_test[:,l]    
    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)    


    # only shuffle X_train part 
    # X_train, y_train =removePlanned(X_train, y_train)
    X_train, y_train=separatePMUs(X_train, y_train)
    # X_train, y_train = shuffle(X_train, y_train)   

    # X_test, y_test=removePlanned(X_test, y_test)
    X_test, y_test= separatePMUs(X_test,y_test)
 
    
    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)    
    
    
    return X_train,y_train,X_test,y_test    

def main():


    # ck = '2016-'+str(num)
    list1 = []
    list2 = []
    for m in range(0,4):    
        X_train,y_train,X_val, y_val= read_data(m)
        validation_set = (X_val, y_val)

        string = '2016-'+str(m)+"best_model_so_far"

        model = load_model(str(string)+'.h5')  
        y_pred=model.predict_classes(X_val) 
        

        locals()['p'+str(m)] = pd.DataFrame(y_val)
        locals()['p'+str(m)].columns = ['real']
        locals()['p'+str(m)]['pred']= y_pred
        locals()['p'+str(m)].to_csv('label_'+str(m)+'.csv',index = None)
    

        
if __name__ == "__main__":
    main()





# scores = model.evaluate(X_val, y_val, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))




