import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from gen_dis import run 

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

def data_pack(q1,q2,d1,d2,turb,option,L,ind,i):
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
    path = "gene/"
    up_f = path+"up_"+str(turb)+"_opt_"+str(option)+"_"+str(ind)
    down_f =path+"down_"+str(turb)+"_opt_"+str(option)+"_"+str(ind)
    # i = 1
    if option == 1:
        nor_f = "seasonal/"+str(option)+"0m/gene/winner_"+turb+"a"+str(ind)
    else:
        nor_f = "seasonal/"+str(option)+"0m/gene/"+turb+"-a"+str(ind)
        
        
    pred_nor = tt_neat(test_nor_X,test_nor_y,nor_f)
    # pred_up = run_neat(tr_up_X,tr_up_y,test_up_X,test_up_y,iter_num,up_f)
    pred_up = tt_neat(test_up_X,test_up_y,up_f)
    # pred_down = run_neat(tr_down_X,tr_down_y,test_down_X,test_down_y,iter_num,down_f)
    pred_down = tt_neat(test_down_X,test_down_y,down_f)
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
    return dt.values
    

def gene_index(option):

    tr = pd.read_csv("../../data/ppmit12_2009.csv")    
    test = pd.read_csv("../../data/ppmit12_2010.csv")   
    max_y = 221

       
    
    if option == 1:
        tr = tr.loc[:,['ws','detadir','l7','l6','l5','l4','l3','l2','l1','l0']]
        test = test.loc[:,['ws','detadir','l7','l6','l5','l4','l3','l2','l1','l0']]
        valp = 7
        valn = 5
        
    elif option == 2:
        tr = tr.loc[:,['ws','detadir','l8','l7','l6','l5','l4','l3','l2','l0']]
        test = test.loc[:,['ws','detadir','l8','l7','l6','l5','l4','l3','l2','l0']]
        valp = 7
        valn = 7
        
    elif option == 3:
        tr = tr.loc[:,['ws','detadir','l9','l8','l7','l6','l5','l4','l3','l0']]
        test = test.loc[:,['ws','detadir','l9','l8','l7','l6','l5','l4','l3','l0']]
        # valp = 5
        # valn = 10
        valp = 10
        valn = 10
    elif option == 4:
        tr =     tr.loc[:,['ws','detadir','l10','l9','l8','l7','l6','l5','l4','l0']]
        test = test.loc[:,['ws','detadir','l10','l9','l8','l7','l6','l5','l4','l0']]
        valp = 7
        valn = 7
        
    elif option == 5:
        tr =    tr.loc[:,['ws','detadir','l11','l10','l9','l8','l7','l6','l5','l0']]
        test = test.loc[:,['ws','detadir','l11','l10','l9','l8','l7','l6','l5','l0']]
        valp = 10
        valn = 10
        
    elif option == 6:
        tr = tr.loc[:,['ws','detadir','l12','l11','l10','l9','l8','l7','l6','l0']]
        test = test.loc[:,['ws','detadir','l12','l11','l10','l9','l8','l7','l6','l0']]
        valp = 5
        valn = 10    

    # print(tr.iloc[:, [-5, -2]])
    tr['diff'] = tr.iloc[:, -5]-tr.iloc[:, -2]
    test['diff'] = test.iloc[:, -5]-test.iloc[:,-2]
    # test['diff'] = test['l4']-test['l1']
    



    df1 = test[(test['diff']>0.01*valp*max_y)&(test['diff']<0.15*max_y)]
    df2 = test[(test['diff']<-0.01*valn*max_y)&(test['diff']>-0.15*max_y)]
    df3 = test[(test['diff']>0.15*max_y)|(test['diff']<-0.15*max_y)|(test['diff']>-0.01*valn*max_y)&(test['diff']<0.01*valp*max_y)]
    
    print(df1.shape)
    
    print(df1.head())
    print(df2.shape)
    print(df2.tail())
    

    df = pd.concat([df1, df2])


    print(df.head())
    print(df.shape)
    print(df.tail())   
    
    idx = df.index.tolist()
    
    return idx


    

    
def get_para(turb,option):
    if(turb =='mit'):
        if(option == 1):
            d = 534
            q = 98
            L = 3
            # {'target': 0.5306224275598818, 'params': {'L': 3.0, 'd1': 533.6037777617784, 'q1': 98.0}}
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
            d = 586
            q = 40
            L = 4
        elif(option == 6):
            d = 560
            q =40
            L =5



    elif(turb =='ge'):
        
        if(option == 1):
            d = 800
            q = 75
            L = 3
            # {'target': 0.3954272721472926, 'params': {'L': 3.0, 'd1': 800.0, 'q1': 74.49221767299947}}
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
            d = 441
            q = 40
            L = 4

        elif(option == 6):
            d = 560
            q =40
            L =5
            
    return L,d,q
    
def cc1(option):
    turb = 'mit'
    # option = 3
    
    L,d1,q1 = get_para(turb,option)
    
    q2 = q1
    d2 =d1
    for i in range(1,7):
        ind = i
        pred = data_pack(q1,q2,d1,d2,turb,option,L,ind,i)
        pred = pd.DataFrame(pred)
        pred.to_csv("data/"+turb+"_"+str(i)+".csv",index = None)

def cc2(option):
    turb = 'ge'
    # option = 3
    
    L,d1,q1 = get_para(turb,option)
    
    q2 = q1
    d2 =d1
    for i in range(1,7):
        ind = i
        pred = data_pack(q1,q2,d1,d2,turb,option,L,ind,i)
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
    option = 4

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


 