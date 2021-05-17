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
from gene_pred import gene_pred
from neat_pred import neat_pred
from rev import fun3

def data_pack(q1,q2,d1,d2,turb,option,L,ind):
    year = 2008
    #runing Generations for Neat
    iter_num = 400
    # turb = 'mit'
    n =1500
    # q = 95
    # d = 500
    show = 0
    tr_up_ind,test_up_ind =fun3(q1,d1,turb,0,option,L)
    tr_down_ind,test_down_ind =fun3(q1,d1,turb,1,option,L)
    listA = list(range(144*365))
    
    retC = list(set(tr_up_ind).union(set(tr_down_ind)))
    tr_nor_ind = [i for i in listA if i not in retC]

    retD = list(set(test_up_ind).union(set(test_down_ind)))
    test_nor_ind = [i for i in listA if i not in retD]    
    
    # print(len(tr_nor_ind))
    # print(len(test_nor_ind))    
    
    if turb == 'mit':
        tr = pd.read_csv("../../../data/ppmit12_2009.csv")    
        test = pd.read_csv("../../../data/ppmit12_2010.csv") 
        max_y = 221
    elif turb == 'ge':
        tr = pd.read_csv("../../../data/ppge12_2009.csv")    
        test = pd.read_csv("../../../data/ppge12_2010.csv")  
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
    path = "../gene/"
    up_f = path+"up_"+str(turb)+"_opt_"+str(option)+"_"+str(ind)
    down_f =path+"down_"+str(turb)+"_opt_"+str(option)+"_"+str(ind)
    
    pred_nor = neat_pred(test_nor_X,test_nor_y,turb,option)
    
    pred_up = gene_pred(test_up_X,test_up_y,turb,option,ind,1)
    
    # gene_pred(test_up_X,test_up_y,turb,option)
    # run_neat(tr_up_X,tr_up_y,test_up_X,test_up_y,iter_num,up_f)
    

    pred_down = gene_pred(test_down_X,test_down_y,turb,option,ind,0)
    # run_neat(tr_down_X,tr_down_y,test_down_X,test_down_y,iter_num,down_f)
    
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

    # print("ind="+str(ind)+" "+str(turb)+" NMAE=",100*mae/max_y)
    return real, dt.values
    


def get_para(turb,option):
    if(turb =='mit'):
        if(option == 1):
            d = 534
            q = 98
            L = 3
            
        elif(option == 2):
            d = 262
            q = 40
            L = 4
            
        elif(option == 3):
            d = 282
            q = 41
            L = 3
            
        elif(option == 4):
            d = 664
            q = 42
            L = 3
            
        elif(option == 5):
            d = 741
            q = 40
            L = 3

        elif(option == 6):
            d = 560
            q =40
            L =5



    elif(turb =='ge'):
    
        if(option == 1):
            d = 800
            q = 74
            L = 3
            
        elif(option == 2):
            d = 531
            q = 42
            L = 3
        
        elif(option == 3):
            d = 741
            q = 40
            L = 3

        elif(option == 4):
            d = 741
            q = 40
            L = 3

        elif(option == 5):
            d = 741
            q = 40
            L = 3


        elif(option == 6):
            d = 560
            q =40
            L =5
            
    return L,d,q
    

    
def get_best(option):
    mit =0
    ge =0
    if option ==1:
        mit = 9
        ge =9
        
    elif option ==2:
        mit = 7
        ge =10               
        
    elif option ==3:
        mit = 2
        ge =10        
    elif option ==4:
        mit = 10
        ge =10       
    elif option ==5:
        mit = 1
        ge =10             
        
    elif option ==6:
        mit = 2
        ge =10       
    return mit,ge
    
def get_combine(option):

    turb = 'ge'
    L,d1,q1 = get_para(turb,option)
    q2 = q1
    d2 =d1


    turb = 'mit'
    L2,d,q = get_para(turb,option)
    q3 = q
    d3 =d   

    mit,ge = get_best(option)

    
    

    r2, p2 = data_pack(q,q3,d,d3,'mit',option,L2,mit) 

    r1, p1 = data_pack(q1,q2,d1,d2,'ge',option,L,ge) 
  
        

    r1=r1+r2
    p1=p1+p2
    p1 = pd.DataFrame(p1)
    p1.to_csv("DSN_"+str(option)+"0m.csv",index = None)
    print("DSN_"+str(option)+"0m saved")

    
    
def main():

    s1 = timeit.default_timer()  
    get_combine(2)
    get_combine(5)
    s2 = timeit.default_timer()  
    print ('Runing time is Hour:',round((s2 -s1)/3600,2))
    
""" 
"""

if __name__ == "__main__":
    main()


 
