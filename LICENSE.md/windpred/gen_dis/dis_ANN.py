import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras

def get_data(turb,option):

    
    if turb == 'mit':
        tr = pd.read_csv("../../data/ppmit12_2009.csv")    
        test = pd.read_csv("../../data/ppmit12_2010.csv") 
        max_y = 221
    elif turb == 'ge':
        tr = pd.read_csv("../../data/ppge12_2009.csv")    
        test = pd.read_csv("../../data/ppge12_2010.csv")  
        max_y = 53*1.5
        
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
    
    return tr,test,max_y




def cc1(option):

    turb = "mit"
    tr,test,max_y = get_data(turb,option)

    y_test = test.pop("l0")
    X_test = test
    path2 = "seasonal/ann/"
    for i in range(1,6+1):
        model = keras.models.load_model(path2 + str(option)+"-"+turb+"-"+str(i))
        pred = model.predict(X_test)
        mae= mean_absolute_error(pred.flatten(),y_test.values)
        print(str(turb)+" i= "+str(i)+" NMAE=",100*mae/max_y)
        pred = pd.DataFrame(pred)
        pred.to_csv("data/"+turb+"_"+str(i)+".csv",index = None)

def cc2(option):

    turb = "ge"
    tr,test,max_y = get_data(turb,option)

    y_test = test.pop("l0")
    X_test = test
    path2 = "seasonal/ann/"
    for i in range(1,6+1):
        model = keras.models.load_model(path2 + str(option)+"-"+turb+"-"+str(i))
        pred = model.predict(X_test)
        mae= mean_absolute_error(pred.flatten(),y_test.values)
        print(str(turb)+" i= "+str(i)+" NMAE=",100*mae/max_y)
        pred = pd.DataFrame(pred)
        pred.to_csv("data/"+turb+"_"+str(i)+".csv",index = None)

def deal():

    for i in range(1,6+1):
        df1 = pd.read_csv("data/ge_"+str(i)+".csv")
        for j in range(1,6+1):
            df2 = pd.read_csv("data/mit_"+str(j)+".csv")
            tt = df1.values + df2.values
            print(tt.shape)
            tt = pd.DataFrame(tt)
            tt.to_csv("data/tt_"+str(j)+".csv",index = None)

from gen_dis import run 
def main():
    option = 6
    cc1(option)
    cc2(option)
    deal()
    run(option)

if __name__ == "__main__":
    main()


 