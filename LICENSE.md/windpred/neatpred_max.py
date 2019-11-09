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
def grimshaw(yt):

    ymean = sum(yt)/float(len(yt))
    ymin = min(yt)
    xstar = 2*(ymean - ymin)/(ymin)**2
    total = 0 
    for i in yt:
            total = total + math.log(1+xstar*i)
    vx = 1 + (1/len(yt))*total
    gam =vx -1
    sig = gam/float(xstar)          
    return gam,sig

def calT(q,gam,sig,n,no_t,t):

    zq = t+ (sig/gam)*(((q*n/no_t)**(-gam))-1)  
    return zq
    
    
def pot_func(x,q):

    
    t = np.percentile(x,q)
    nt = [n for n in x if n>t]
    yt = [n-t for n in nt]
    ymean = sum(yt)/float(len(yt))
    ymin = min(yt)
    xstar = 2*(ymean - ymin)/(ymin)**2

    total = 0
    no_t = len(nt)
    n = len(x)
    # gam,sig<--Grimshaw(yt)
    for i in yt:
        total = total + math.log(1+xstar*i)

    vx = 1 + (1/len(nt))*total
    gam =vx -1
    sig = gam/float(xstar)
    
    
    
    
    
    # zq<--calcthereshold(q,... n,nt,t)
    
    zq = t+ (sig/gam)*(((q*n/no_t)**(-gam))-1)   #function(1)
    return zq,t
    # print ("Inital Threshold", t)
    # print ("Updated Threshold", zq)
    # print ("len nt = ", len(nt))
    # print ("len yt = ", len(yt))

# from IPython.core.pylabtools import figsize

"""
"""

# print(data.shape)


def eval_fun(X_train,y_train,X_test,y_test):


    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config')

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(100))
    
    # Run until a solution is found.
    winner = p.run(eval_genomes, 1)  # run for 12000 to test 
    # with open('winner_ge'+str(i), 'wb') as f:
        # pickle.dump(winner, f)
    
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
    
    yhat = pred
    y= y_test
    mae= mean_absolute_error(y, yhat)

    max_y = 221
    
    return 100*mae/max_y

# input 
# n: lens of calibration data
# d: window size
# q: quantile
def fun1(q,d,n,show,year,turb):

    i = 0 #initial point
    df1 = pd.read_csv("../data/total_"+str(turb)+"_"+str(year)+".csv")    
    data = pd.read_csv("../data/total_"+str(turb)+"_"+str(year+1)+".csv")

    df1[df1<0] = 0
    data[data<0] = 0
    # print(df1.shape)
    df1 = df1.iloc[-(d+n):]
    # print(df1.shape)
    data = pd.concat([df1, data], axis=0)
    data=data.reset_index(drop = True)
    # print(df1.head())
    # print(data.head())

    # print('data.shape',data.shape)

    df = data.loc[0:d]
    x = data.values

    # print(len(x))


    # x = np.arange(1,52060+1)

    # figsize(16, 4)
    # print(df.shape)
    # df2 = data.loc[d+1:d+n]
    # print(df2.shape)
    # df3 = data.loc[d+n+1:]
    # print(df3.shape)
    n2 = len(x)
    

    M = np.zeros(n2+2,float)
    y = np.zeros(n2+2,float)

    wstar = df.values

    M[d+1] = np.mean(wstar)
    xp = np.zeros(n2,float)


    list = []
    for i in range(d+1,d+n):
        xp[i] = x[i]-M[i]
        wstar = x[i-d+1:i]
        M[i+1] = np.mean(wstar)
        list.append(M[i+1])
        
    # print(len(list))
    zq,t = pot_func(xp[d+1:d+n],q)
    # print("zq",zq)
    # print("t",t)

    zzq =zq*np.ones(n2)
    # A is a set of anomalies
    Avalue = []
    Aindex = []

    k = n
    k2 = len(x)-n-d
    yt = []
    no_t = 0
    result = []
    for i in range(d+n,d+n+k2):
        xp[i] = x[i] - M[i]
        # print("xp =",xp[i])
        if xp[i]>zq:
            # print("yeah1")
            Avalue.append(x[i])
            Aindex.append(i)
            M[i+1] = M[i]

        elif xp[i]>t:
            print("yeah2")
            y[i] = xp[i]-t
            yt.append(y[i])
            no_t = no_t +1
            k = k+1
            gam,sig = grimshaw(yt)
            zq = calT(q,gam,sig,k,no_t,t)
            wstar =np.append(wstar[1:],x[i])
            M[i+1] = np.mean(wstar)
            zzq[i+1] = zq
            # result.append(zq)
        else:
            # print("yeah3")
            k = k+1
            wstar =np.append(wstar[1:],x[i])
            M[i+1] = np.mean(wstar)
        # print(M[i+1]) 
    # print(result)     
    line = np.arange(1,n2+1)        
    # print(len(zzq))
    # print(len(x))
    # print(len(xp))

    data['d'] = 0
    data.loc[Aindex,'d'] = 1
    # print("Anomaly case number = ",len(Avalue))
    # print(data.head(20))
    print(data['d'].mean())
    data =  data.iloc[(n+d):]
    # print(data.shape)
    label1 = data['d']

    label1 = label1.reset_index(drop=True)
    
    # print('label1 shape:',label1.shape)
    # data.to_csv("../data/label_"+str(turb)+str(year+1)+".csv",header = 1,index = None)
    # print(str(turb)+str(year+1)+' save done!')



    return label1
    # print(df2.iloc[2,(df2.shape[0]-1)]+ df2.iloc[3,(df2.shape[0]-1)])
    # print(type(df2))
    # print(SS)
    
    
# This will get the result of running neats
from run_neat import run_neat
from neat_pred import neat_pred

def data_pack(q1,q2,d1,d2,turb):
    year = 2008
    iter_num = 100
    # turb = 'mit'
    n =1500
    # q = 95
    # d = 500
    show = 0
    df1 = fun1(q1,d1,n,show,year,turb)  
    df2 = fun1(q2,d2,n,show,year+1,turb)  
    tr  = pd.read_csv('../data/pp'+str(turb)+'12_'+str(year+1)+'.csv')
    test  = pd.read_csv('../data/pp'+str(turb)+'12_'+str(year+2)+'.csv')
    tr = tr.iloc[:,[0,1,7,8,9,10,11,12,13,14]]
    test= test.iloc[:,[0,1,7,8,9,10,11,12,13,14]]
    tr = pd.concat([tr, df1], axis=1,join_axes=[tr.index])
    test = pd.concat([test, df2], axis=1,join_axes=[test.index])
    
    df5 = tr[tr['d']==0]
    df6 = test[test['d']==0]
    
    
    
    
    df3 = tr[tr['d']==1]
    df4 = test[test['d']==1]
    df3.pop('d')
    df4.pop('d')
    df6.pop('d')
    
    y_train = df3.pop('l0')
    y_test =  df4.pop('l0')
    ddf6 =  df6.pop('l0')
    X_train = df3
    X_test = df4
    # print(df3.head())
    # print(df4.head())
    print(df3.shape)
    print(df4.shape)
    print('df6.shape = ',df6.shape)
    
    X_train = X_train.values.tolist()
    y_train = y_train.values.tolist()
    X_test = X_test.values.tolist()
    y_test = y_test.values.tolist()
    
    p1 = run_neat(X_train,y_train,X_test,y_test,iter_num)
    print(p1.shape)
    
    df6 = df6.values.tolist()
    ddf6 = ddf6.values.tolist()
    p2 = neat_pred(df6,ddf6)
    print(p2.shape)
    max_y = 221
    yhat = np.append(p1,p2)
    

    y = y_test+ddf6
    mae= mean_absolute_error(y, yhat)

    print("Mits NMAE=",100*mae/max_y)
    return (1/mae)
    
from bayes_opt import BayesianOptimization

def black_box_function(q1,q2,d1,d2):
    qq1 = q1
    qq2 = q2
    dd1 = int(round(d1))
    dd2 = int(round(d2))
    turb = 'mit'

    return data_pack(qq1,qq2,dd1,dd2,turb)


def main():

    # data_pack(70,70,200,200,'mit')

    s1 = timeit.default_timer()  
    # Bounded region of parameter space
    pbounds = {'q1': (70, 99), 'd1': (200, 800),
                'q2': (70, 99), 'd2': (200, 800)
                }

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=3,
        n_iter=7
    )

    print(optimizer.max)        
    s2 = timeit.default_timer()  
    print ('Runing time is Hour:',round((s2 -s1)/3600,2))
    
""" 
"""

if __name__ == "__main__":
    main()

    
    
""" 

    # year can be only 2008 or 2009
    year = 2009
    # turb can be only ge or mit
    turb = 'ge'
    q = 90.55
    d = round(500.5)
    n =1500
    show = 0
    print('sum of corr is ',fun1(q,d,n,show,year,turb))

"""     
