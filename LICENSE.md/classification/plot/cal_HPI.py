# Tocalulate HPI value based on FFT result
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt
from scipy.fftpack import fft


import os
import sys
import timeit
import pickle


    




def rd2(k,i):
    path1 = '../pickleset2/'

    # list = ['ip_m','vp_a','va_m','va_a','vb_m','vb_a','vc_m','vc_a','rocof']
    list = ['rocof','vp_m','ip_m','ip_a']

    p1 = open(path1 +'X_S'+str(k)+'_'+str(list[i])+'_6.pickle',"rb")

    pk3  = pd.read_csv(path1 +'y_S'+str(k)+'_'+str(list[2])+'_6.csv')
    
    # pk3 = pk3[:,1]   
    # pk3 = pk3.astype(np.int32)      

    pk1 = pickle.load(p1)

    X_train = pk1
    y_train = pk3.values
    # X_train, y_train  = rm(X_train, y_train)


    return X_train, y_train    
    
def date_fft(y,j,label):
    path = "gen/stat/"
    num = 360
    n=y.shape[0]# 信号长度
    t = range(n)
    yy=fft(y)
    yf=abs(yy)#取绝对值
    yf1=(fft(y))/n#归一化处理
    
    # print(yf1.shape)
    yf2=yf1[range(int(n/num))]##由于对称性，只取一半区间

    xf=np.arange(len(y))#频率
    xf1=xf
    xf2=xf[range(int(n/num))]#取一半区间

    max = np.max(np.real(yf2))
    print(max)
    sum = 0
    
    for i in range(0,30):
        sum +=np.real(yf2[i])
    print(sum)
    
    print("ratio: " + str(round(sum/max,2)))
    
    #显示原始序列
    plt.figure()
    plt.subplot(211)
    plt.plot(t,y,'g')
    plt.xlabel("Time")
    plt.ylabel("Amplitude ip_a")
    plt.title("Original wave label ="+str(label))

    # 显示归一化处理后单边序列
    plt.subplot(212)
    plt.plot(xf2, yf2, 'b')
    plt.xlabel("Freq (Hz)")
    plt.ylabel("Y(freq)")
    plt.title("max:"+str(round(max,1))+" ratio: "+str(round(sum/max,2)) ,fontsize=10,color='#F08080')
    plt.tight_layout()
    plt.savefig(path+'label = '+str(label)+"_"+str(j))
    
def cal2(y):    
    num = 360
    n=y.shape[0]# 信号长度
    t = range(n)
    yy=fft(y)
    yf1=(fft(y))/n#归一化处理
    yf2=yf1[range(int(n/num))]
    max = np.max(np.real(yf2))
    sum = 0
    for i in range(0,30):
        sum +=np.real(yf2[i])
    return round(sum/max,2)
        
def cal(df):
    list2 =[]
    for j in range(df.shape[0]):
        list = []
        for i in range(23):
            temp = df[j,i,:]
            list.append(cal2(temp) )
        temp2 =  np.array(list)
        list2.append(round(np.mean(temp2),1))
    print(list2)        
            
def pt(label):  
    list = ['y1-','c2-','g+','.-','r-3','b-4']
    path = "gen/stat/"
    size = (13,5)
    m1 = np.zeros(size)
    m2 = np.zeros(size)
    m3 = np.zeros(size)
    plt.figure(figsize=(8,6))
    for ii in range(1,2):
        X_train, y_train = rd2(ii,3)
        # print(type(np.where(y_train==3)[0]))
        a = np.where(y_train==label)[0]

        if(a.shape[0]>0):
            df = X_train[a]

            df =  np.squeeze(df)
            # df = df.reshape(-1,10800)
            cal(df)
            # for i in range(0,23):

                # date_fft(df[i],i,label)


    





def main(cc):
    s1 = timeit.default_timer()  

    pt(cc)
    
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
if __name__ == '__main__':  
    cc = int(sys.argv[1])
    main(cc)
