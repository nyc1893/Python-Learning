import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import sys
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
from neat_pred import neat_pred
from rev2 import fun2,fun3

def data_pack(q1,q2,d1,L,turb,option,ind):
    year = 2008
    #runing Generations for Neat
    iter_num = 100
    # turb = 'mit'
    n =1500
    # q = 95
    # d = 500
    # L =3
    

    
    show = 0
    tr_up_ind,test_up_ind =fun2(q1,d1,turb,option,L)
    tr_down_ind,test_down_ind =fun3(q2,d1,turb,option,L)
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


    path = "gene2/"
    up_f = path+"up_"+str(turb)+"_opt_"+str(option)+"_"+str(ind)
    down_f =path+"down_"+str(turb)+"_opt_"+str(option)+"_"+str(ind)
    
    pred_nor = neat_pred(test_nor_X,test_nor_y,turb,option)
    pred_up = run_neat(tr_up_X,tr_up_y,test_up_X,test_up_y,iter_num,up_f)
    pred_down = run_neat(tr_down_X,tr_down_y,test_down_X,test_down_y,iter_num,down_f)
    
    
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

    tt = "q = "+str(q1)+",d = "+str(d1)+" NMAE="+str(100*mae/max_y)
    print(tt)

    with open('grid_search/log2.txt','a+') as f:    #设置文件对象
        f.write(tt+'\n')                 #将字符串写入文件中
    

    


def main():
    # data_pack(q1,q2,d1,d2,turb,option)
    # data_pack(q1,q2,d1,d2,turb,option):
    # data_pack(70,70,200,200,'mit',1)

    s1 = timeit.default_timer()  
        # {'Mit': 0.5302329082719374, 'params': {'d1': 400, 'q1': 95, 'q2': 87  }}
    # {'GE': 0.3970207105485811, 'params': {'d1': 500.0, 'q1': 95.0, 'q2': 60.0}}
    
    # q1 = 95
    # q2 = 87
    # d1 = 400
    # turb = "mit"
    
    q1 = int(sys.argv[1])
    q2 = q1
    # d1 = int(sys.argv[2])
    turb = "mit"
    # ll = [400,500,600]
    ll = [700,800]
    # j = 1000
    option = 1
    L = 6
    for j in ll:
        for i in range(1,3+1):
            data_pack(q1,q2,j,L,turb,option,i)


    s2 = timeit.default_timer()  
    print ('Runing time is Hour:',round((s2 -s1)/3600,2))
    
    
    
# cd py_file/wind/code2/pred

if __name__ == "__main__":
    main()
