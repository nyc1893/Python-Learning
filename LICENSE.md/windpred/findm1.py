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
from unsave_neat import run_neat
from neat_pred import neat_pred
from rev import fun3

def data_pack(q1,q2,d1,d2,turb,option,L):
    year = 2008
    #runing Generations for Neat
    iter_num = 100
    # turb = 'mit'
    n =1500
    # q = 95
    # d = 500
    show = 0
    tr_up_ind,test_up_ind =fun3(q1,d1,turb,0,option,L)
    tr_down_ind,test_down_ind =fun3(q1,d1,turb,2,option,L)
    listA = list(range(144*365))
    
    retC = list(set(tr_up_ind).union(set(tr_down_ind)))
    tr_nor_ind = [i for i in listA if i not in retC]

    retD = list(set(test_up_ind).union(set(test_down_ind)))
    test_nor_ind = [i for i in listA if i not in retD]    
    
    print(len(tr_nor_ind))
    print(len(test_nor_ind))    
    
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
    
    
    tr_up_X = tr.iloc[tr_up_ind]
    tr_up_y = tr_up_X.pop("l0")
    
    tr_down_X = tr.iloc[tr_down_ind]
    tr_down_y = tr_down_X.pop("l0")    

    # df4 = tr.iloc[tr_nor_ind]
    # dff4 = df4.pop("l0")   

    real = test["l0"].values
    test_up_X = test.iloc[test_up_ind]
    test_up_y = test_up_X.pop("l0")
    
    test_down_X = test.iloc[test_down_ind]
    test_down_y = test_down_X.pop("l0")    
    
    test_nor_X = test.iloc[test_nor_ind]
    test_nor_y = test_nor_X.pop("l0")   

    pred_nor = neat_pred(test_nor_X,test_nor_y,turb,option)
    pred_up = run_neat(tr_up_X,tr_up_y,test_up_X,test_up_y,iter_num)
    pred_down = run_neat(tr_down_X,tr_down_y,test_down_X,test_down_y,iter_num)
    
    pred_nor = pd.DataFrame(pred_nor)
    pred_up = pd.DataFrame(pred_up)
    pred_down = pd.DataFrame(pred_down)
    
    pred_nor.index = test_nor_ind
    pred_up.index =  test_up_ind
    pred_down.index = test_down_ind
    
    
    dt  =  pd.concat([pred_nor,pred_up,pred_down])
    dt.sort_index(inplace=True)
    # print(dt.head())
    # print(dt.tail())
    # print(dt.shape)    
    
    mae= mean_absolute_error(real, dt.values)

    print(str(turb)+" NMAE=",100*mae/max_y)
    return (1/(100*mae/max_y))
    

    
from bayes_opt import BayesianOptimization

def black_box_function(q1,d1,L):
    qq1 = int(round(q1))
    qq2 = qq1
    dd1 = int(round(d1))
    dd2 = dd1
    L2 = int(round(L))
    turb = 'mit'

    return data_pack(qq1,qq2,dd1,dd2,turb,1,L2)


def main():
    # data_pack(q1,q2,d1,d2,turb,option)
    # data_pack(q1,q2,d1,d2,turb,option):
    # data_pack(70,70,200,200,'mit',1)

    s1 = timeit.default_timer()  
    
    
 
    # Bounded region of parameter space
    pbounds = {'q1': (10, 40), 'd1': (100, 800),
                'L':(3,10)
                }

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=50,
        n_iter=50
    )
    tt = str(optimizer.max)
    with open('Mit3_gap_1.txt','a+') as f:    #设置文件对象
        f.write(tt+'\n')                 #将字符串写入文件中
    print(tt)        
    s2 = timeit.default_timer()  
    print ('Runing time is Hour:',round((s2 -s1)/3600,2))
    
    
    
""" 
Mit2 and all change rev threshold from 0.1Pmax to 0.03Pmax
Mit2 -- searching space:  q(40-98)
Mit3 -- searching space:  q(10-40)

"""

if __name__ == "__main__":
    main()


 
