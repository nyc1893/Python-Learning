import numpy as np
import pandas as pd
import os
import neat
import pickle

# Should for ramp pred
            
def run_neat(X_train,y_train,X_test,y_test,iter_num,filename):        
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config')
    X_train = X_train.values.tolist()
    y_train = y_train.values.tolist()                         
                         
    X_test = X_test.values.tolist()
    y_test = y_test.values.tolist()
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
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(100))
    
    # Run until a solution is found.
    winner = p.run(eval_genomes, iter_num)  # run for 12000 to test 
    with open(filename, 'wb') as f:
        pickle.dump(winner, f)
    
    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    # Make and show prediction on unseen data (test set) using winner NN's 
    # genome.
    print('\nOutput:')
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-49')
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
