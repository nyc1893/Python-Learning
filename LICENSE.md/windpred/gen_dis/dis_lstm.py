
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM

import pandas as pd
import numpy as np
import timeit

from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# from keras.layers import Activation, Dense, BatchNormalization, TimeDistributed
from keras.models import load_model

import os


def get_result(turb,option,path):
    path2= "../../data/"
    if turb == 'mit':
        tr = pd.read_csv(path2+ "ppmit12_2009.csv")    
        test = pd.read_csv(path2+ "ppmit12_2010.csv")  
        # max_y = 221
    elif turb == 'ge':
        tr = pd.read_csv(path2+ "ppge12_2009.csv")    
        test = pd.read_csv(path2+ "ppge12_2010.csv")  
        # max_y = 1.5*53
        
    if option == 1:
        tr = tr.loc[:,['ws','detadir','l7','l6','l5','l4','l3','l2','l1','l0']]
        test = test.loc[:,['ws','detadir','l7','l6','l5','l4','l3','l2','l1','l0']]

    elif option == 2:
        tr = tr.loc[:,['ws','detadir','l8','l7','l6','l5','l4','l3','l2','l0']]
        test = test.loc[:,['ws','detadir','l8','l7','l6','l5','l4','l3','l2','l0']]

    elif option == 3:
        tr = tr.loc[:,['ws','detadir','l9','l8','l7','l6','l5','l4','l3','l0']]
        test = test.loc[:,['ws','detadir','l9','l8','l7','l6','l5','l4','l3','l0']]
        
    elif option == 4:
        tr = tr.loc[:,['ws','detadir','l10','l9','l8','l7','l6','l5','l4','l0']]
        test = test.loc[:,['ws','detadir','l10','l9','l8','l7','l6','l5','l4','l0']]

    elif option == 5:
        tr = tr.loc[:,['ws','detadir','l11','l10','l9','l8','l7','l6','l5','l0']]
        test = test.loc[:,['ws','detadir','l11','l10','l9','l8','l7','l6','l5','l0']]

    elif option == 6:
        tr = tr.loc[:,['ws','detadir','l12','l11','l10','l9','l8','l7','l6','l0']]
        test = test.loc[:,['ws','detadir','l12','l11','l10','l9','l8','l7','l6','l0']]

    # print(tr.head())
    # print(test.head())

    train_y =  tr.pop('l0')
    test_y =  test.pop('l0')

    train_y = train_y.values
    test_y = test_y.values

    train_X = tr.values
    test_X = test.values


    train_X = train_X[:,np.newaxis,:]
    test_X = test_X[:,np.newaxis,:]


    y = test_y

    model = load_model(path)
    yhat = model.predict(test_X)
    # print(yhat.shape)
    return  yhat,y




list_select = [2,4,5,6,16,20]

def cc1(option):
    turb = 'mit'
    max_y = 221
    print(turb)
    # for i in range(1,22+1):
    v = 1
    for i in list_select:
        path1 = 'lstm/result'+str(option)+'/'+turb+'-'+str(i)
        y1,a = get_result(turb,option,path1)
        pred = pd.DataFrame(y1)
        pred.to_csv("data/"+turb+"_"+str(v)+".csv",index = None)
        v+=1
        mae= mean_absolute_error(y1,a)
        # print("i = "+str(i)+", all NMAE=",100*mae/max_y)


def cc2(option):
    turb = 'ge'
    max_y = 79.5
    print(turb)
    v = 1
    for i in list_select:
        path1 = 'lstm/result'+str(option)+'/'+turb+'-'+str(i)
        y1,a = get_result(turb,option,path1)
        pred = pd.DataFrame(y1)
        pred.to_csv("data/"+turb+"_"+str(v)+".csv",index = None)
        v+=1
        mae= mean_absolute_error(y1,a)
        # print("i = "+str(i)+", all NMAE=",100*mae/max_y)
        

def deal():

    for i in range(1,6+1):
        df1 = pd.read_csv("data/ge_"+str(i)+".csv")
        for j in range(1,6+1):
            df2 = pd.read_csv("data/mit_"+str(j)+".csv")
            tt = df1.values + df2.values
            # print(tt.shape)
            tt = pd.DataFrame(tt)
            tt.to_csv("data/tt_"+str(j)+".csv",index = None)
            
from gen_dis import run 
def main():
    s1 = timeit.default_timer()  
    # option = 3
    ll = [1,3,4,5,6]
    for option in ll:
        print("option:"+str(option))
        cc1(option)
        cc2(option)
        deal()
        run(option)

    s2 = timeit.default_timer()  
    print ('Runing time is min:',round((s2 -s1)/60,2))
    
    
if __name__ == "__main__":
    main()
    