

# import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import math
import pandas as pd
import time  
from sklearn import metrics  
import pickle as pickle  
import timeit
import matplotlib.pyplot as plt
from scipy.linalg import svd 
# import time
import timeit
from random import randint

from sklearn.metrics.pairwise import cosine_similarity


def pic(ii):

    path1 = '../../../pickleset2/'

    df1 = pd.read_csv("data/Ss"+str(ii)+".csv")
    df2 = pd.read_csv("data/Sub"+str(ii)+".csv")

    dt = df2[df2["label"]==0 ]
    dt["dif_line"] = dt["line_ind"] - dt["linesub_ind"] 
    dt = dt[dt["dif_line"]>0]
    dt.replace([np.inf,-np.inf],np.nan)
    # dt.dropna(axis=0, how='any')
    dt.fillna(-1)
    ind = dt.index
    dt2 = df1.iloc[ind]


    dt.pop("line_ind")
    dt.pop("freq_ind")
    dt.pop("linesub_ind")
    dt.pop("freqsub_ind")
    dt3 = dt.pop("dif_line")
    dt.pop("label")
    
    dt2.pop("label")
    dt2.pop("word")
    dt2.pop("word2")
    dt2.pop("season")
    dt2.pop("No")

    return dt,dt2,dt3
    
def fun():
    p1,k1,m1 = pic(1)
    for ii in range(2,13+1):
        p2,k2,m2 = pic(ii)
        p1 = pd.concat([p1, p2])
        k1 = pd.concat([k1, k2])
        m1 = pd.concat([m1, m2])
    p1.fillna(-1)
    k1.fillna(-1)
    print(p1.shape)
    print(k1.shape)
    p1 = p1.values
    k1 = k1.values
    m1 = m1.values
    res =[]
    time = []
    i = 0
    # print(p1[i])
    # print(k1[i])
    for i in range(p1.shape[0]):
        if(i!=-1):
            print(i)

            cc=cosine_similarity(p1[i].reshape(1,-1),k1[i].reshape(1,-1))

            res.append(cc[0][0])
            time.append(m1[i])
    df = pd.DataFrame(res)
    df.columns = ['sim']
    df["dif_time"] = time
    df["dif_time"] = df["dif_time"]/60
    print(df.head())
    df.to_csv("data/line.csv",index = 0)
    
def main():
    s1 = timeit.default_timer()  
    
    fun()

    s2 = timeit.default_timer()
    print('Time:(min) ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

