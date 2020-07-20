


import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import math
import pandas as pd
from datetime import datetime, timedelta
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import InputLayer, Input
# from tensorflow.keras.layers import Reshape, MaxPooling2D
# from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
# from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import load_model
# from tensorflow.keras import optimizers,regularizers

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import sys
# import skopt
# from skopt import gp_minimize, forest_minimize
# from skopt.space import Real, Categorical, Integer
# from skopt.plots import plot_convergence
# from skopt.plots import plot_objective, plot_evaluations
# from skopt.utils import use_named_args

import pickle
import timeit
import datetime

start = timeit.default_timer()

def recover(df):
    dd = np.zeros(df.shape[0])
    dd = dd.astype("int8") 
    rr = df[df['L2']>0].index.tolist()
    dd[rr] = 1
    
    rr = df[df['L3']>0].index.tolist()
    dd[rr] = 2    


    rr = df[df['L4']>0].index.tolist()
    dd[rr] = 3    
    
    df['rev'] = dd
    return df
    
def evaluate(train_y,pred):
    
    # train_y = df['v'].values
    # pred = df[str(name)].values
    # a = train_y.shape
    matrix=confusion_matrix(train_y, pred)
    print(matrix)
    class_report=classification_report(train_y, pred)
    print(class_report)
    # print(str(name))    

def read_ll():  

    list1 =[]
    list2 =[]
    # list1 = [[]for i in range(4)]
    # list2 = [[]for i in range(4)]
    for i in range(1,4+1):
        df= pd.read_csv('label_'+str(i-1)+'.csv')
        list1.append(df['real'])
        list2.append(df['pred'])
        
    dr = pd.DataFrame(list1).values.transpose()
    dp = pd.DataFrame(list2).values.transpose()
    
    dr = pd.DataFrame(dr)
    dp = pd.DataFrame(dp)
    
    dr.columns = ['L1','L2','L3','L4']
    dp.columns = ['L1','L2','L3','L4']
    
    dr = recover(dr)
    dp = recover(dp)
    evaluate(dr['rev'],dp['rev'])
    
def main():
    read_ll()

    
    

if __name__ == '__main__':  
    # global best_accuracy 
    main()   
    
