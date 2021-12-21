
# get the result and save to csv file

import numpy as np
import pandas as pd
import os
import neat
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from functools import reduce 
import operator
import timeit
from gen_dis import run 


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
    
def get_real(turb,option,season):  
    if turb == 'mit':
        tr = pd.read_csv("../../data/ppmit12_2009.csv")    
        test = pd.read_csv("../../data/ppmit12_2010.csv")   
        max_y = 221
    elif turb == 'ge':
        tr = pd.read_csv("../../data/ppge12_2009.csv")    
        test = pd.read_csv("../../data/ppge12_2010.csv")  
        max_y = 53*1.5

        
    if season == 1:
        ind1 = 0
        ind2 = 13140

    elif season == 2:
        ind1 = 13140
        ind2 = 13140*2
 
    elif season == 3:    
        ind1 = 13140*2
        ind2 = 13140*3

    elif season == 4:
        ind1 = 13140*3 
        ind2 = 13140*4 
    test = test.iloc[ind1:ind2]    
    y3 = test.pop('l0').values
    return max_y,y3    
    
    
def data_pack(turb,option,season):

    
    if turb == 'mit':
        tr = pd.read_csv("../../data/ppmit12_2009.csv")    
        test = pd.read_csv("../../data/ppmit12_2010.csv")   
        max_y = 221
    elif turb == 'ge':
        tr = pd.read_csv("../../data/ppge12_2009.csv")    
        test = pd.read_csv("../../data/ppge12_2010.csv")  
        max_y = 53*1.5

        
    if season == 1:
        ind1 = 0
        ind2 = 13140
        # path = 'gene-positive2/'+str(turb)+'-'
    elif season == 2:
        ind1 = 13140
        ind2 = 13140*2
        # path = 'gene-negative2/'+str(turb)+'-'        
    elif season == 3:    
        ind1 = 13140*2
        ind2 = 13140*3
        # path = 'gene3/'+str(turb)+'-'   
    elif season == 4:
        ind1 = 13140*3 
        ind2 = 13140*4 
        # path = 'gene4/'+str(turb)+'-'   
    
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
        tr =     tr.loc[:,['ws','detadir','l10','l9','l8','l7','l6','l5','l4','l0']]
        test = test.loc[:,['ws','detadir','l10','l9','l8','l7','l6','l5','l4','l0']]

    elif option == 5:
        tr =    tr.loc[:,['ws','detadir','l11','l10','l9','l8','l7','l6','l5','l0']]
        test = test.loc[:,['ws','detadir','l11','l10','l9','l8','l7','l6','l5','l0']]

    elif option == 6:
        tr = tr.loc[:,['ws','detadir','l12','l11','l10','l9','l8','l7','l6','l0']]
        test = test.loc[:,['ws','detadir','l12','l11','l10','l9','l8','l7','l6','l0']]
    

    test = test.iloc[ind1:ind2]

    y1 = test.pop('l0')

    
    df1 = test.values.tolist()

    
    y1 = y1.values.tolist()

    return df1,y1

def get_result(turb,option,season,i):

    mit_normal = np.array([[7,1,7,5],[5,2,3,6],[3,2,7,6],[9,9,9,6],[5,4,3,6],[1,3,2,1]])
    ge_normal = np.array([[10,3,6,1],[3,2,1,5],[3,3,3,5],[1,5,2,1],[3,2,3,1],[2,2,2,2]])

    ramp_arr = np.array([[9,2,2,3],[2,1,2,3],[1,5,4,2],[5,5,4,3],[1,1,1,2],[2,5,5,5]])


    # print(ramp_arr[option-1][0])
    if (turb=='mit'):

        path3 = 'seasonal/'+str(option)+'0m/gene'+str(season)+'/'+str(turb)+'-'+str(season)+'-'+str(i)
        # print("value = ",mit_arr[season-1])
    else:

        path3 = 'seasonal/'+str(option)+'0m/gene'+str(season)+'/'+str(turb)+'-'+str(season)+'-'+str(i)
        
        
    df1,y1 = data_pack(turb,option,season)
    
    yhat1 = get_pred(df1,y1,path3)

    
    # print(yhat1.shape)
    # print(yhat2.shape)
    # print(yhat3.shape)


    df = pd.DataFrame(yhat1)

    df = df.values
    return df
    
def cc1(option):
    turb = 'mit'
    # list1 =[]
    print(turb)
    for i in range(1,6+1):
        list1 =[]
        for season in range(1,4+1):
            result = get_result(turb,option,season,i)
            list1.append(result.tolist())
            
        pred = np.array(reduce(operator.concat, list1))  
        pred = pd.DataFrame(pred)

        pred.to_csv("data/"+turb+"_"+str(i)+".csv",index = None)
        
def cc2(option):
    turb = 'ge'
    # list1 =[]
    print(turb)
    for i in range(1,6+1):
        list1 =[]
        for season in range(1,4+1):
            result = get_result(turb,option,season,i)
            list1.append(result.tolist())
            
        pred = np.array(reduce(operator.concat, list1))  
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
    
""" 
"""

if __name__ == "__main__":
    main()
    
    
