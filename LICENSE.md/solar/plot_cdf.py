#plot pic just as what Prof. ask
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# from test_neat import test_neat
import time
from datetime import datetime
import statsmodels.api as sm # recommended import according to the docs
from scipy.stats import genpareto


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

# This will get the result of running neats
# from run_neat import run_neat
# from neat_pred import neat_pred


# input 
# n: lens of calibration data
# d: window size
# q: quantile
def fun1(q,d,n):

    i = 0 #initial point
    # df1 = pd.read_csv("../data/total_"+str(turb)+"_"+str(year)+".csv")    
    data2 = pd.read_csv("solar1.csv")
    # print(data.head())
    # data2 = data2.loc[0:500]
    data = data2['power'].values
    data = pd.DataFrame(data)


    df = data.loc[0:d]
    x = -data.values
    x2 = -data.values
    # x2 = data2['power'].values
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
    # print(data)

    data.loc[Aindex,'d'] = 1

    data =  data.iloc[(n+d):]
    # print(data.shape)
    show = 0
    if show ==1:
        # line = np.arange(1,n2+1)
        plt.figure(figsize = (24,4))
        plt.plot(line,x2,'b--',label='power curve')
        # plt.plot(line,xp,color='blue')
        # plt.scatter(line,M[0:n2],color='green')
        xais = np.arange(1,len(Avalue)+1)		
        plt.scatter(Aindex,x2[Aindex],color='red',label='extreme points')
        plt.title('window size=%s q=%s Lca=%s Anomaly number=%s'%(d,q,n,len(Avalue)), fontsize=10)	
        plt.suptitle('solar power-2013')
        plt.xlabel('time (1hour)')
        plt.ylabel('solar power')
        plt.legend(loc='lower left')
        plt.text(5, 5, t, ha='right', rotation=-15, wrap=True)		
        plt.plot(Aindex,x[Aindex],color='red')		
 
        # plt.savefig(str(name)+"dspot-q="+str(q), dpi=150)	    
        plt.show()
    
    label1 = data['d']
    label1 = label1.reset_index(drop=True)    
    # print(label1.shape)
    # print(label1.mean())
    return label1
        
def extract_time (df,name):
    dt =  df[name]
    dt = dt.apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df["year"] = dt.map(lambda x: x.year)
    df["month"] = dt.map(lambda x: x.month)
    df["day"] = dt.map(lambda x: x.day)
    df["hour"] = dt.map(lambda x: x.hour)    
    return df   
    
    
def data_pack2(id,name,start,s1,s2):
    val = 0.5
    
    data = pd.read_csv("solar"+str(id)+".csv")    
    data = extract_time (data,'time')
    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index('time')
    dt = data[start]
    
    # print(dt.head())
    dt = dt[dt['hour']>=s1]
    dt = dt[dt['hour']<s2]
    dt = dt[name]
    # print(dt.shape)
    # return dt
    return dt.values
    
def data_pack(id,name,start,q,d,n,flag,s1,s2):
    val = 0.5
    
    data = pd.read_csv("solar"+str(id)+".csv")    
    # data = data.iloc[(n+d):]
    df = fun1(q,d,n,name)
    # df = pd.DataFrame(df)
    # print(df.head())
    data['d'] = df

    data = extract_time (data,'time')
    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index('time')
    dt = data[start]
    
    # print(dt.head())
    dt = dt[dt['hour']>=s1]
    dt = dt[dt['hour']<s2]
    # if flag ==1:
        # dt = dt[dt['d']>val]
    # elif flag ==0:
        # dt = dt[dt['d']<val]
        
    
    # dt = dt[dt['power']>0]
    # dt = dt[name]
    print(dt.shape)
    return dt
    # return dt.values
    
    
def data_pack3(id,name,start,q,d,n,flag):
    val = 0.5
    
    data = pd.read_csv("solar"+str(id)+".csv")    
    # data = data.iloc[(n+d):]
    df = fun1(q,d,n)
    # df = pd.DataFrame(df)
    # print(df.head())
    data['d'] = df

    # data = extract_time (data,'time')
    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index('time')
    dt = data[start]
    dt.loc[dt['power'] == 0, 'd'] = 0
    dt = dt[dt[name]>0]
    if flag ==1:
        dt = dt[dt['d']>val]
    elif flag ==0:
        dt = dt[dt['d']<val]
    dt = dt[name]    
    # print(dt.head(50))
    print(dt.shape)
    return dt
    
    
    # return dt.values
def plot1(id,name,i):
    
    # data_pack2(id,name,start,s1,s2)
    plt.figure(figsize = (6,4))
    
    list = ['6:00-8:00','9:00-11:00','12:00-14:00','15:00-17:00','18:00-20:00'] 
    # for i in range(0,1):
    for j in range(0,4+1):
        sample = data_pack2(id,name,'2013Q'+str(i+1),6+j*3,9+j*3)
        ecdf = sm.distributions.ECDF(sample)
        x = np.linspace(0, max(sample))
        y = ecdf(x)
        plt.plot(x, y, linewidth = '1',label= list[j])


    plt.xlabel('solar '+ str(name) + ' Ramps')
    plt.ylabel('CDF')
    plt.title('CDF of solar '+str(name)+' of 2013 season '+str(i+1)+' in location '+str(id))
    # plt.suptitle('window size=%s q=%s Lca=%s'%(d,q,n), fontsize=10)
    # plt.legend(bbox_to_anchor=(1,1))
    plt.legend(loc='lower right')
    # plt.legend(loc='upper left')
    
    plt.savefig("wp_s"+str(i), dpi=150)

    # plt.show()   

    
# this is to plot non-ramp and ramp CDF
def plot2(id,name,n,q,d):
    
    
    plt.figure(figsize = (4,3))

    start = '2013'              
    sample = data_pack3(id,name,start,q,d,n,1)  
    ecdf = sm.distributions.ECDF(sample)
    x = np.linspace(0, max(sample))
    y = ecdf(x)
    plt.plot(x, y,'r', linewidth = '1',label='ramp')

    sample = data_pack3(id,name,start,q,d,n,0)  
    ecdf = sm.distributions.ECDF(sample)
    x = np.linspace(0, max(sample))
    y = ecdf(x)
    plt.plot(x, y, 'y',linewidth = '1',label='non-ramp')
    
    c = 0.03
    mean, var, skew, kurt = genpareto.stats(c, moments='mvsk')


# sample = np.linspace(genpareto.ppf(0.01, c),genpareto.ppf(0.99, c), 100)
    # x = np.linspace(0,1, 100)
    x = np.linspace(genpareto.ppf(0.01, c),genpareto.ppf(0.99, c), 500)
    sample =  genpareto.pdf(x, c)            
    ecdf = sm.distributions.ECDF(sample)
    x = np.linspace(0, max(sample))
    y = ecdf(x)
    plt.plot(x, y, 'b.',label='Theoretical GPD' )  
    
    
    plt.xlabel('Normalized solar '+ str(name))
    plt.ylabel('CDF')
    # plt.title('CDF of solar '+str(name)+' of 2013 in location '+str(id))
    # plt.suptitle('window size=%s q=%s Lca=%s'%(d,q,n), fontsize=10)
    # plt.legend(bbox_to_anchor=(1,1))
    
    
    
    plt.legend(loc='lower right')
    # plt.legend(loc='upper left')
    
    
    plt.tight_layout()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(ls='--')
    plt.savefig('ramp-nonramp', dpi=300)
    plt.show()   



   
# this is to plot hourly and seasonly CDF    
def plot3(id,name):
    
    # data_pack2(id,name,start,s1,s2)
    plt.figure(figsize = (5,3))
    # plt.figure(figsize = (6,3))
    list = ['r-1','g2-','b-3','y+-','-4']
    # list = ['-1','2-','-3','+-','-4']
    list2  = ['6:00-8:00','9:00-11:00','12:00-14:00','15:00-17:00','18:00-20:00'] 
    i = 0
    for j in range(0,3+1):
        sample = data_pack2(id,name,'2013Q'+str(i+1),6+j*3,9+j*3)
        ecdf = sm.distributions.ECDF(sample)
        x = np.linspace(0, max(sample))
        y = ecdf(x)
        plt.plot(x, y, str(list[j]),linewidth = '1',label=str(list2[j]))


    i = i+2
    for j in range(0,3+1):
        sample = data_pack2(id,name,'2013Q'+str(i+1),6+j*3,9+j*3)
        ecdf = sm.distributions.ECDF(sample)
        x = np.linspace(0, max(sample))
        y = ecdf(x)
        # plt.plot(x, y, str(list[j]),linewidth = '1',label='winter '+str(list2[j]))
        plt.plot(x, y, str(list[j]),linewidth = '1')
    
    
    s1 = 'summer'
    s2 = 'winter'
    xd = 0.37
    yd = 0.63
    color = 'black'
    lenwidth = 0.3
    hw = 5
    hl =7
    plt.annotate(s2, xy=(0.5, 0.5), xytext=(xd, yd),
            xycoords='data',
            arrowprops=dict(facecolor= color, shrink=0.05,
            width = lenwidth,headwidth = hw,headlength = hl)
            )
            
    plt.annotate(s2, xy=(0.6, 0.5), xytext=(xd, yd),
            xycoords='data',
            arrowprops=dict(facecolor= color, shrink=0.05,
            width = lenwidth,headwidth = hw,headlength = hl)
            )
            
    plt.annotate(s2, xy=(0.4, 0.96), xytext=(xd, yd),
            xycoords='data',
            arrowprops=dict(facecolor= color, shrink=0.05,
            width = lenwidth,headwidth = hw,headlength = hl)
            )
            
    plt.annotate(s2, xy=(0.37, 0.82), xytext=(xd, yd),
            xycoords='data',
            arrowprops=dict(facecolor= color, shrink=0.05,
            width = lenwidth,headwidth = hw,headlength = hl)
            )
            
            
            
    xd = 0.6
    yd = 0.9
    color = 'white'
    lenwidth = 0.3
    plt.annotate(s1, xy=(0.47, 0.95), xytext=(xd, yd),
            xycoords='data',
            arrowprops=dict(facecolor= color, shrink=0.05,
            width = lenwidth,headwidth = hw,headlength = hl)
            )
            
    plt.annotate(s1, xy=(0.49, 0.88), xytext=(xd, yd),
            xycoords='data',
            arrowprops=dict(facecolor= color, shrink=0.05,
            width = lenwidth,headwidth = hw,headlength = hl)
            )
            
    plt.annotate(s1, xy=(0.76, 0.76), xytext=(xd, yd),
            xycoords='data',
            arrowprops=dict(facecolor= color, shrink=0.05,
            width = lenwidth,headwidth = hw,headlength = hl)
            )
            
                          
    plt.xlabel('Normalized solar '+ str(name))
    plt.ylabel('CDF')
    # plt.title('CDF of solar '+str(name)+' of 2013 in location '+str(id))
    # plt.suptitle('window size=%s q=%s Lca=%s'%(d,q,n), fontsize=10)
    # plt.legend(bbox_to_anchor=(1,1))
    plt.legend(loc='lower right')
    # plt.legend(loc='upper left')
    # plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.) 
    
    plt.tight_layout()
   
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(ls='--')
    plt.savefig("hourly", dpi=300)
    plt.show()   

# This is to find the most sunny day for smart persistence 
def find_day(id,name,month):

    list = [31,28,31,30,31,30,31,31,30,31,30,31]
    data = pd.read_csv("solar"+str(id)+".csv")    
    data = extract_time(data,'time')
    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index('time')
    max_value = 0


    ind = 1
    for i in range (1,9+1):
        dt = data['2013-'+str(month)+'-'+str(i)]
        if(sum(dt['power'].values)>max_value):
            max_value = sum(dt['power'].values)
            ind = i
    for i in range (10,list[month-1]):
        dt = data['2013-'+str(month)+'-'+str(i)]
        if(sum(dt['power'].values)>max_value):
            max_value = sum(dt['power'].values)
            ind = i
        
    # print('max_value',max_value)
    # print('ind = ', ind)
    # print(list)
    return ind
    
def record(id,name):
    list = []
    for i in range(1,13):
        a = find_day(id,name,i)
        list.append(a)
    # print(list)
    return list

def proc(id,name,month):
    list = record(id,name)
    data = pd.read_csv("solar"+str(id)+".csv")    
    data = extract_time(data,'time')
    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index('time') 
    list2 = [31,28,31,30,31,30,31,31,30,31,30,31]
    
    i = 1
    dt = data['2013-'+str(month)+'-'+str(i)]    
    dd = data['2013-'+str(month)+'-'+str(list[0])]  
    k = dd['power'].values - dt['power'].values
    for i in range(2,9+1):
        dt = data['2013-'+str(month)+'-'+str(i)]   
        k2 = dd['power'].values - dt['power'].values
        k = np.append(k,k2)
    # print(list2[month-1])
    for i in range(10,list2[month-1]+1):
        dt = data['2013-'+str(month)+'-'+str(i)]   
        k2 = dd['power'].values - dt['power'].values        
        k = np.append(k,k2)
    print(k.shape[0]/24)   
    return k
    # print(list[0])
    
    
def save_k():
    q = 80
    n = 24*7
    d = 24
    id =1
    i = 1
    name = 'power'
    k  = proc(id,name,i)
    for i in range(2,13):
        k2  = proc(id,name,i)
        k = np.append(k,k2)
    k = pd.DataFrame(k)
    # print(k.head(50))
    k.to_csv("k.csv", index = None)


    
def main():
    q = 70
    n = 24*7
    d = 24
    id =1
    i = 1
    name = 'power'
    start = '2013'
    flag = 1
    # data_pack3(id,name,start,q,d,n,flag)
    # plot2(id,name,n,q,d)
    plot3(id,name)
    
    
if __name__ == "__main__":
    main()
    
    
    
    
