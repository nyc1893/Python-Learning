from pyexpat.errors import XML_ERROR_TAG_MISMATCH, XML_ERROR_TEXT_DECL
import numpy as np
import pandas as pd
import os
import neat
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import timeit
def data_pack():
    tr = pd.read_csv("solar_training.csv") 
    test = pd.read_csv("solar_test.csv")
    test = test.dropna()
    #tr.dropna()
   # test.dropna()

    y_train = tr.pop('POWER')
    y_test =  test.pop('POWER')
    tr.pop("TIMESTAMP")
    test.pop("TIMESTAMP")
    X_train = tr
    X_test = test
    # print(tr.tail())
    # print(X_train.shape)
    # print(tr.dtypes)
    # print(test.dtypes)
    # print(X_test.shape)
    # print(tr.shape)

    X_test = X_test.values.tolist()
    y_test = y_test
    pred = get_pred(X_test,y_test,"gene/aa1")
    return pred, y_test


def get_pred(X_test,y_test,path):        
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config')
    
    f=open(path,'rb')  
    winner=pickle.load(f)  
	

    list2 = []
    # print ('type(list2)',type(list2))

    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    for xi, xo in zip(X_test, y_test):
        output = winner_net.activate(xi)
        # print ('type(output)',type(output))
        list2.append(output)
        # print("  input {!r}, expected output {!r}, got {!r}".format(
        # xi, xo, output))
    pred = np.array(list2)
    return(pred)
    
    
def fun2():
    pred,y = data_pack()
    
    test = pd.read_csv("solar_test.csv")
    test = test.dropna()
    y =  test.pop('POWER').values

    pred = np.squeeze(pred)
    max_y = 1
    print(pred.shape)
    print(y.shape)
    
    print ("Number is nan ",len(pred[np.isnan(pred)]))
    print ("Number is nan ",len(y[np.isnan(y)]))
    
    rmse = sqrt(mean_squared_error(pred, y))
    mae= mean_absolute_error(pred, y)
    print(" NMAE=",100*mae/max_y)
    print(" NRMSE=",100*rmse/max_y)
    
def fun3():
    df = pd.read_csv("solar_test.csv")
    print(df.shape)
    num = df.isna().sum()
    print(num)
    
    
def persistant():
    df = pd.read_csv("solar_test.csv")
    print(df.shape)
    df["y2"] = df["POWER"].shift(1)
    df = df.dropna()
    y = df["POWER"].values
    pred = df["y2"].values
    print(df.head())
    print(df.shape)
    max_y = 1
    rmse = sqrt(mean_squared_error(pred, y))
    mae= mean_absolute_error(pred, y)
    print(" NMAE=",100*mae/max_y)
    print(" NRMSE=",100*rmse/max_y)    
    
    
def main():
    s1 = timeit.default_timer()  

    # fun2()
    persistant()

    s2 = timeit.default_timer()  
    print ('Runing time is Hour:',round((s2 -s1)/3600,2))
    
"""
"""

if __name__ == "__main__":
    main()
    
    
