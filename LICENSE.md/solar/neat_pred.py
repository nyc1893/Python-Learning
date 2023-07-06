import numpy as np
import pandas as pd
import os
import neat
import pickle


def neat_pred(X_test,y_test,turb,option):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config') 

    
    if option ==1:
        if turb =='ge':              
            name = '/ge-31'
        else:
            name = '/mit-40'
    
    elif option ==2: 
        if turb =='ge':              
            name = '/ge-18'
        else:
            name = '/mit-14'    
            
    elif option ==3: 
        if turb =='ge':              
            name = '/ge-2'
        else:
            name = '/mit-23'    

    elif option ==4: 
        if turb =='ge':              
            name = '/ge-5'
        else:
            name = '/mit-32'    

    elif option ==5: 
        if turb =='ge':              
            name = '/ge-10'
        else:
            name = '/mit-5' 

    elif option ==6: 
        if turb =='ge':              
            name = '/ge-9'
        else:
            name = '/mit-17'                
    
    
    ge_path  = '../data/pred'
    f=open(ge_path+name,'rb')  
    winner2=pickle.load(f)  



    list2 = []
    # print ('type(list2)',type(list2))

    winner_net = neat.nn.RecurrentNetwork.create(winner2, config)
    for xi, xo in zip(X_test, y_test):
        output = winner_net.activate(xi)
        # print ('type(output)',type(output))
        list2.append(output)
        # print("  input {!r}, expected output {!r}, got {!r}".format(
        # xi, xo, output))
    pred = np.array(list2)
    return pred