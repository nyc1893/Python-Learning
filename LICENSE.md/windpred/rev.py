import math
import pandas as pd
import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt
from scipy.stats import genpareto

# delta p are used 
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


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

    # The paper is 98%  have nothing to do with the q
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

# input 
# n: lens of calibration data
# d: window size
# q: quantile
def fun1(n,d,turb,sign,L):
    # L = 5
    i = 0 #initial point
    df1 = pd.read_csv("../../../data/total_"+str(turb)+"_2008.csv")    
    df2 = pd.read_csv("../../../data/total_"+str(turb)+"_2009.csv")    
    df3 = pd.read_csv("../../../data/total_"+str(turb)+"_2010.csv") 
    
    df1[df1<0] = 0
    df2[df2<0] = 0    
    df3[df3<0] = 0
    # df1 = df1.iloc[-(d+n):]
    
    # print("2008 shape",df1.shape)
    data = pd.concat([df1, df2], axis=0)
    data = pd.concat([data, df3], axis=0)

    t1 = data.values
    num = t1.shape[0] - L
    cc = np.zeros(num)
    dd = np.zeros(num)     
    for i in range(num):
        cc[i] = t1[i+L]-t1[i]
        dd[i] = t1[i]-t1[i+L]

    # print(data.shape)
    cc = pd.DataFrame(cc)
    # print(cc.head())
    # print(cc.shape)  
    
    dd = pd.DataFrame(dd)
    # print(dd.head())  
    # print(dd.tail())  
    # print(dd.shape)     
    cc.columns = ['a']
    dd.columns = ['a']
    
    if turb =='ge':              
        max_y = 53*1.5
     
        
    else:
        max_y = 221
        
    temp = dd[dd["a"]>0.1*max_y]
    ind = temp.index.tolist()
    temp2 = cc[cc["a"]>0.1*max_y]
    ind2 = temp2.index.tolist()  
    
    if sign ==0:              
        return temp2,ind2
     

    return temp,ind  
    
    

    
def fun3(q,d,turb,sign,offset,L):
    n = 1500
    data,ind = fun1(n,d,turb,sign,L)
    
    
    # now x is delta power
    x = data.values


    n2 = len(x)
    

    M = np.zeros(n2+2,float)
    y = np.zeros(n2+2,float)

    # wstar = df.values

    # M[d+1] = np.mean(wstar)
    xp = np.zeros(n2,float)


    list = []

    zq,t = pot_func(x[d+1:d+n],q)

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

        if x[i]>zq:
            # print("yeah1")
            Avalue.append(x[i])
            Aindex.append(i)
            # M[i+1] = M[i]

        elif x[i]>t:
            print("yeah2")
            y[i] = xp[i]-t
            yt.append(y[i])
            no_t = no_t +1
            k = k+1
            gam,sig = grimshaw(yt)
            zq = calT(q,gam,sig,k,no_t,t)
            # wstar =np.append(wstar[1:],x[i])
            # M[i+1] = np.mean(wstar)
            zzq[i+1] = zq

        else:

            k = k+1

    # print(len(Avalue))
    # print(len(Aindex))
    Aindex = np.array(Aindex)
    # print(ind[:5])
    # print(ind[-5:])
    ck1 = []
    ck2 = []
    for i in range(len(Aindex)):
        if ind[Aindex[i]]<52560+10091-offset and ind[Aindex[i]]>10091-offset:
            ck1.append(ind[Aindex[i]]-10091+offset)
        elif ind[Aindex[i]]>52560+10091-offset :
            ck2.append(ind[Aindex[i]]-10091-52560+offset)
    # print(x[ind[Aindex[0]]])
    # print(ck1[:5])
    # print(ck1[-5:])
    # print(ck2[:5])
    # print(ck2[-5:])
    # ck1 : 2009 
    # ck2 : 2010
    
    return ck1,ck2
    



def main():
    # plot_m()
    # fun1(10,30,"mit")
    fun3(20,200,1,"mit")
""" 
"""

if __name__ == "__main__":
    main()
    
    
    
    
