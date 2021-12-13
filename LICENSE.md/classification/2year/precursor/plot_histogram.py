

import numpy as np
import math
import pandas as pd
import time  
import datetime
import matplotlib.pyplot as plt


font = {'family':'Times New roman'  #'serif', 
#         ,'style':'italic'
        ,'weight':'normal'  # 'bold' 
#         ,'color':'red'
        ,'size':13
       }

font2 = {'family':'Times New roman'  #'serif', 
#         ,'style':'italic'
        ,'weight':'normal'  # 'normal' 
#         ,'color':'red'
        ,'size':13
       }    
       
def run_16():   
    # x = np.random.randn(10000)
    df = pd.read_csv("osc.csv")
    # df2 = pd.read_csv("L_5.csv")
    # df = pd.concat([df, df2])
    x = df["dif_time"].values
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot()
    ax.hist(x, bins=50, color='blue', alpha=0.7)
    #刻度值字体设置
    labels = ax.get_xticklabels()+ ax.get_yticklabels()
    [label.set_fontname('Times New roman') for label in labels]
    plt.tick_params(labelsize=13)
    plt.xlabel('Time ahead(Seconds)',font2) 
    plt.ylabel('Counts',font) 
    plt.grid(ls='--')
    plt.tight_layout()
    plt.show()
    
    
def plt_scatter():   
    # x = np.random.randn(10000)
    df = pd.read_csv("line.csv")
    df2 = pd.read_csv("L_4.csv")
    df = pd.concat([df, df2])
    # x = df["dif_time"].values
    fig = plt.figure(figsize=(5,3))
    # ax = fig.add_subplot(figsize=(5,3))
    ax = df.plot.scatter(x = "dif_time",y = "sim" )
    #刻度值字体设置
    labels = ax.get_xticklabels()+ ax.get_yticklabels()
    [label.set_fontname('Times New roman') for label in labels]
    plt.tick_params(labelsize=13)
    plt.xlabel('Time ahead(Seconds)',font2) 
    plt.ylabel('Similarity',font) 
    plt.grid(ls='--')
    plt.tight_layout()
    plt.show()
    
    
def main():   

    run_16()
    # plt_scatter()
if __name__ == '__main__':  

    main()
    # update_complete_random(0.5)
