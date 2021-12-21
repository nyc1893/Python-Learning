import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def eval_genomes(genomes, config): #function Used for training model 
# using the training set
    for genome_id, genome in genomes:
        genome.fitness = -1
        net = neat.nn.RecurrentNetwork.create(genome, config)
        for xi, xo in zip(X_train, y_train):
            output = net.activate(xi)
            # xo = list(xo)
            # print(type(output))
            # print(type(xo))
            genome.fitness -= (output[0] - xo) ** 2 #Distance from 
            # the correct output summed for all 84 inputs patterns

# This will get the result of running neats
from run_neat import run_neat
from neat_predd import neat_predd,tt_neat
from rev import fun3
import sys

def data_pack(turb,option,i):

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
    
   
    real = test["l0"].values
    
    test_nor_X = test
    test_nor_y = test_nor_X.pop("l0")       
    if option == 1:
        nor_f = "seasonal/"+str(option)+"0m/gene/winner_"+turb+"a"+str(i)
    else:
        nor_f = "seasonal/"+str(option)+"0m/gene/"+turb+"-a"+str(i)
        
        
    pred_nor = tt_neat(test_nor_X,test_nor_y,nor_f)    
    # pred_nor = neat_predd(test_nor_X,test_nor_y,turb,i)
    dt = pd.DataFrame(pred_nor)
    
    # dt  =  pred_nor
   
    
    mae= mean_absolute_error(real, dt.values)

    print(str(turb)+" NMAE=",100*mae/max_y)
    return dt.values
    


def cc1(option):
    turb = 'mit'

    for i in range(1,7):
        ind = i
        pred = data_pack(turb,option,i)
        pred = pd.DataFrame(pred)
        pred.to_csv("data/"+turb+"_"+str(i)+".csv",index = None)

def cc2(option):
    turb = 'ge'

    for i in range(1,7):

        pred = data_pack(turb,option,i)
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

    s1 = timeit.default_timer()  
    option = 6

    cc1(option)
    cc2(option)
    deal()
    run(option)
    s2 = timeit.default_timer()  
    print ('Runing time is mins:',round((s2 -s1)/60,2))
    
""" 
"""

if __name__ == "__main__":
    main()


 