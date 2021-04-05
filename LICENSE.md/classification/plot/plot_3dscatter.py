
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import timeit
import datetime
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D 
import os
import sys

def check(ii):
    path = "freq/"
    df = pd.read_csv(path+"savedft_23_"+str(ii)+".csv")
    df = df.dropna()
    df.to_csv(path+"savedft_23_"+str(ii)+".csv",index = None)
    
def ff():

    path = "freq/"
    df2 = pd.read_csv(path+"savedft_23_1.csv")
    
    for i in range(2,14):
        df = pd.read_csv(path+"savedft_23_"+str(i)+".csv")
        df2 = pd.concat([df2,df])
        
    ind0 = df2[df2["23"]==0].index    
    ind1 = df2[df2["23"]==1].index
    ind2 = df2[df2["23"]==2].index
    ind3 = df2[df2["23"]==3].index
    ind4 = df2[df2["23"]==4].index
    ind5 = df2[df2["23"]==5].index    
    
    
    # print(ind0)
    # print(ind1)
    # print(ind2)
    # print(ind3)
    # df4 = df2[df2["23"]!=3]
    df2.pop("23")
    temp = df2
    df2["avg"] = temp.mean(axis=1)
    df2["std"] = temp.std(axis=1)
    df2["mm"] = temp.median(axis=1)
    # temp = df4
    # df4["avg"] = temp.mean(axis=1)
    # df4["std"] = temp.std(axis=1)    
    
    # plt.scatter(df2.loc[ind0,['avg']], df2.loc[ind0,['std']], color='green', label='Line')
    # plt.scatter(df2.loc[ind1,['avg']], df2.loc[ind1,['std']], color='blue', label='Trans')
    # plt.scatter(df2.loc[ind2,['avg']], df2.loc[ind2,['std']], color='red', label='Freq')
    # plt.scatter(df2.loc[ind3,['avg']], df2.loc[ind3,['std']], color='black', label='Osc')
    # plt.scatter(df3['avg'], df3['std'], color='blue', label='Osc')
    # plt.xlabel("Avg")
    # plt.ylabel("Std")    
    
    
    # plt.legend(loc='best')
    # plt.show()
    
# def cc():
    # plt.scatter(df4['upBar'], df4['downBar'],df4['std'], color='green', label='Non-freq')
    # plt.scatter(df3['upBar'], df3['downBar'],df3['std'], color='blue', label='freq')
    

    # 绘制散点图
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(df2.loc[ind0,['avg']], df2.loc[ind0,['std']],df2.loc[ind0,['mm']] ,s=0.16, color='c', label='UnplanedLine')
    ax.scatter(df2.loc[ind1,['avg']], df2.loc[ind1,['std']],df2.loc[ind1,['mm']] ,s=0.16, color='g', label='UnplanedTrans')
    ax.scatter(df2.loc[ind2,['avg']], df2.loc[ind2,['std']],df2.loc[ind2,['mm']] ,s=0.16, color='b', label='Freq')
    ax.scatter(df2.loc[ind3,['avg']], df2.loc[ind3,['std']],df2.loc[ind3,['mm']] ,s=0.16, color='r', label='Osc')
    ax.scatter(df2.loc[ind4,['avg']], df2.loc[ind4,['std']],df2.loc[ind4,['mm']] ,s=0.16, color='m', label='planedLine')
    ax.scatter(df2.loc[ind5,['avg']], df2.loc[ind5,['std']],df2.loc[ind5,['mm']] ,s=0.16, color='y', label='planedTrans')    
    # ax.scatter(df2.loc[ind0,['avg']], df2.loc[ind0,['std']],df2.loc[ind0,['mm']] , color='green', label='NonOsc')
    # ax.scatter(df2.loc[ind1,['avg']], df2.loc[ind1,['std']],df2.loc[ind1,['mm']] , color='green', label='NonOsc')
    # ax.scatter(df2.loc[ind2,['avg']], df2.loc[ind2,['std']],df2.loc[ind2,['mm']] , color='green', label='NonOsc')
    # ax.scatter(df2.loc[ind3,['avg']], df2.loc[ind3,['std']],df2.loc[ind3,['mm']] , color='red', label='Osc')
    
    # ax.scatter(df3['r'], df3['sim'],df3['std'], color='blue', label='freq')
     
     
    # 绘制图例
    ax.legend(loc='best')
     
     
    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('median', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('std', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('avg', fontdict={'size': 15, 'color': 'red'})
     

    plt.show()

def ff2():


    path = "freq/"
    df2 = pd.read_csv(path+"savedft_23_1.csv")
    
    for i in range(2,14):
        df = pd.read_csv(path+"savedft_23_"+str(i)+".csv")
        df2 = pd.concat([df2,df])
        
    ind0 = df2[df2["23"]==0].index    
    ind1 = df2[df2["23"]==1].index
    ind2 = df2[df2["23"]==2].index
    ind3 = df2[df2["23"]==3].index
    ind4 = df2[df2["23"]==4].index
    ind5 = df2[df2["23"]==5].index    
    

    df2.pop("23")
    temp = df2
    df2["avg"] = temp.mean(axis=1)
    df2["std"] = temp.std(axis=1)
    df2["mm"] = temp.median(axis=1)


    # 绘制散点图
    fig = plt.figure()
    list_label = ["UnplanedLine","UnplanedTrans","Freq","Osc","planedLine","planedTrans"]
    list_color = ["c","g","b","r","m","y"]
    plt.scatter(df2.loc[ind0,['avg']], df2.loc[ind0,['std']], color=list_color[0], label=list_label[0], s=0.1)
    plt.scatter(df2.loc[ind1,['avg']], df2.loc[ind1,['std']], color=list_color[1], label=list_label[1], s=0.1)
    plt.scatter(df2.loc[ind2,['avg']], df2.loc[ind2,['std']], color=list_color[2], label=list_label[2], s=0.1)
    plt.scatter(df2.loc[ind3,['avg']], df2.loc[ind3,['std']], color=list_color[3], label=list_label[3], s=0.1)
    plt.scatter(df2.loc[ind4,['avg']], df2.loc[ind4,['std']], color=list_color[4], label=list_label[4], s=0.1)
    plt.scatter(df2.loc[ind5,['avg']], df2.loc[ind5,['std']], color=list_color[5], label=list_label[5], s=0.1)
    plt.legend(loc='best')
    
    
    plt.ylabel('std')
    plt.xlabel('avg')
    plt.title('sum of FFT feature: vpm')
    plt.show()




def test():
    a = np.array([1,2,3,4,6])
    b = np.zeros(10)
    for i in range(a.shape[0]):
        b[i] = math.cos(a[i])
    # b = math.cos(a)
    print(b)
    print(b.shape)
    
    
def find_err():
    path = "freq/"
    df2 = pd.read_csv(path+"te.csv")
    df3 = pd.read_csv(path+"X_val.csv")
    df3 = df3[["S","No"]]
    df3 = pd.concat([df3,df2],axis =1)
    ind= df3[df3["label"]!=df3["RF"]].index
    df3 = df3.iloc[ind] 
    print(df3.head())
    print(ind)
    df3.to_csv("err.csv",index =None)

def main():
    s1 = timeit.default_timer()  
    # for i in range(1,14):
        # check(i)
    ff2()
    # test()
    # ff()
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
if __name__ == '__main__':  
    # cc = int(sys.argv[1])
    main()
