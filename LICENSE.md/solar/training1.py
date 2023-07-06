import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
import neat
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
from run_neat2 import run_neat2
#from neat import neat_pred

    
def data_pack():
    tr = pd.read_csv("solar_training.csv") 
    test = pd.read_csv("solar_test.csv")
    y_train = tr.pop('POWER')
    y_test =  test.pop('POWER')
    tr.pop("TIMESTAMP")
    test.pop("TIMESTAMP")
    X_train = tr
    X_test = test
    # print(tr.tail())
    print(X_train.shape)
    print(X_test.shape)
    print(tr.shape)

    X_train = X_train.values.tolist()
    y_train = y_train.values.tolist()
    X_test = X_test.values.tolist()
    y_test = y_test.values.tolist()
    iter_num= 20
    i= 2
    a = "aa"
    path= "gene/"
    run_neat2(X_train,y_train,X_test,y_test,iter_num,i,a,path)
  
def main():
    s1 = timeit.default_timer()  
    data_pack()
    s2 = timeit.default_timer()  
    print ('Runing time is Hour:',round((s2 -s1)/3600,2))
    
""" 
"""

if __name__ == "__main__":
    main()

    
