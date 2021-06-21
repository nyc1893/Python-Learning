

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
import statsmodels.api as sm 
def fun():
    
    df1 = pd.read_csv("NonEvent_S1.csv")
    for i in range(2,14):
        df2= pd.read_csv("NonEvent_S"+str(i)+".csv")
        df1 = pd.concat([df1,df2])
        
    df3 = pd.read_csv("Event_S1.csv")
    for i in range(2,14):
        df4= pd.read_csv("Event_S"+str(i)+".csv")
        df3 = pd.concat([df3,df4])

    data_nor = []
    data_eve = []    
    for i in range(1,9):
        data_nor.append(np.mean(df1[str(i)+"avg"].values))
        data_eve.append(np.mean(df3[str(i)+"avg"].values))

    waters = ('vpm','vpa','ipm','ipa','freq','Rocof','Act','React')

    bar_width = 0.3  # 条形宽度
    index_nor = np.arange(len(waters))  # 男生条形图的横坐标
    index_event = index_nor + bar_width  # 女生条形图的横坐标

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_nor,      height=data_nor, width=bar_width, color='b', label='nor')
    plt.bar(index_event,    height=data_eve, width=bar_width, color='g', label='event')

    # plt.legend()  # 显示图例
    plt.xticks(index_nor + bar_width/2, waters)  
    plt.title('mean avg of zeta value') 


    plt.legend(loc= "best")
    plt.savefig("compare3")  
    
def fun2():
    ll = ["Line","Trans","Freq","Osc"]

    df3 = pd.read_csv("Event_S1.csv")
    for i in range(2,14):
        df4= pd.read_csv("Event_S"+str(i)+".csv")
        df3 = pd.concat([df3,df4])
    for i in range(4):
        dt = df3[df3["label"]==i]
        dt = dt["4max"].values
        print(dt.shape)
        sample = dt
        ecdf = sm.distributions.ECDF(sample)
        x = np.linspace(0, max(sample))
        y = ecdf(x)    
        plt.plot(x,y,label= ll[i])
    plt.xlim(0,  max(sample))
    plt.ylim(0, 1)
    word ="ipa"
    plt.title(word) 
    plt.grid(ls='--')    
    
    plt.legend(loc= "best")
    plt.savefig("cdf-"+word)      
    
    
def main():
    s1 = timeit.default_timer()  
    fun2()

    s2 = timeit.default_timer()
    print('Time: ', (s2 - s1)/60 )
if __name__ == '__main__':  

    main()

