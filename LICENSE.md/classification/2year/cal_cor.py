import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import math
import timeit
import heapq

from numpy import array, sign, zeros
from scipy.interpolate import interp1d
import scipy.signal
from scipy.stats import pearsonr,spearmanr,kendalltau
N = 600
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)


# y1 = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# y2 = np.sin(55.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)

y1 = np.array([0.309016989,0.587785244,0.809016985,0.95105651,1,0.951056526,
0.809017016,0.587785287,0.30901704,5.35898E-08,0,0,
0,0,0,0,0,0,
0,0,0,0,0,0,
0,0,0,0,0,0])+10

y2 = np.array([
0.343282816,0.686491368,0.874624132,0.99459642,1.008448609,
1.014252458,0.884609221,0.677632906,0.378334666,0.077878732,
0.050711886,0.066417083,0.088759401,0.005440732,0.04225661,
0.035349939,0.0631196,0.007566056,0.053183895,0.073143706,
0.080285063,0.030110227,0.044781145,0.01875573,0.08373928,
0.04550342,0.038880858,0.040611891,0.046116826,0.087670453,
])

y3 = np.array([
0.343282816,0.686491368,0.874624132,0.99459642,1.008448609,
1.014252458,0.884609221,0.677632906,0.378334666,0.077878732,
0.050711886,0.066417083,0.088759401,0.005440732,0.04225661,
0.035349939,0.0631196,0.007566056,0.053183895,0.073143706,
0.080285063,0.030110227,0.044781145,0.01875573,0.08373928,
0.04550342,0.038880858,0.040611891,0.046116826,0.087670453,
]).reshape(-1,5)

print(y3.shape)
print(y3)

def Modified_Z(data):
    c = 1.4826
    median = np.median(data)
    # print(median)
    # print("median.shape",median.shape)
    dev_med = np.array(data) -median
    # print("dev_med.shape",dev_med.shape)
    mad = np.median(np.abs(dev_med))
    if mad!=0:
        
        z_score = dev_med/(c*mad)
    else : 
        df = pd.DataFrame(data)
        meanAD = df.mad().values
        z_score =  dev_med/(1.253314*meanAD)
        
    return z_score
    


a = np.array([1,2])
b = np.array([5,6])


def general_equation(first_x,first_y,second_x,second_y):
    # 斜截式 y = kx + b 
    A = second_y-first_y
    B = first_x-second_x
    C = second_x * first_y - first_x * second_y
    k = -1 * A / B
    b = -1 * C / B
    return k, b
    
def cal(a,b,n):
    sum_s12 = 0 
    sum_s1s1 = 0
    sum_s2s2 = 0 
    sum_s1 = 0
    sum_s2 = 0   
    # temp1 = 0
    # temp2 = 0
    delta = 0.0001
    
    for i in range(0,n):
        sum_s12+=a[i]*b[i]
        sum_s1+=a[i]
        sum_s2+=b[i]
        sum_s2s2+=b[i]*b[i]
        sum_s1s1+=a[i]*a[i]
        
    temp1 = n*sum_s1s1-sum_s1*sum_s1
    temp2 = n*sum_s2s2-sum_s2*sum_s2  
    if( (temp1>-delta and temp1<delta) or (temp2>-delta and temp2<delta) or (temp1*temp2<=0)):      
        return -10
        
    pxy = (n*sum_s12-sum_s1*sum_s2)/math.sqrt(temp1*temp2)
    return pxy
    
def date_fft(y):
    num = 8
    n=y.shape[0]# 信号长度
    t = range(n)
    yy=fft(y)
    yf=abs(yy)#取绝对值
    yf1=abs(fft(y))/n#归一化处理
    
    
    yf2=yf1[range(int(n/num))]##由于对称性，只取一半区间
    
    print(yf1.shape)
    
    xf=np.arange(len(y))#频率
    xf1=xf
    xf2=xf[range(int(n/num))]#取一半区间

    #显示原始序列
    plt.figure()
    plt.subplot(211)
    plt.plot(t,y,'g')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Original wave")
    """
    #显示取绝对值后的序列
    plt.subplot(222)
    plt.plot(xf, yf)
    plt.xlabel("Freq (Hz)")
    plt.ylabel("|Y(freq)|")
    plt.title("FFT of Mixed wave(two sides frequency range",fontsize=7,color='#7A378B')
    # 注意这里的颜色可以查询颜色代码表

    #显示归一化处理后双边序列
    plt.subplot(223)
    plt.plot(xf1, yf1)
    # 注意这里的颜色可以查询颜色代码表
    plt.xlabel("Freq (Hz)")
    plt.ylabel("|Y(freq)|")
    plt.title('FFT of Mixed wave(Normalized processing)',fontsize=10,color='#F08080')
    """
    # 显示归一化处理后单边序列
    plt.subplot(212)
    plt.bar(xf2, yf2, alpha=0.99, width = 0.88, facecolor = 'blue', edgecolor = 'white', label='one', lw=1)
    # plt.bar(xf2, yf2, 'b')
    # 注意这里的颜色可以查询颜色代码表
    plt.xlabel("Freq (Hz)")
    plt.ylabel("|Y(freq)|")
    plt.title('FFT of Mixed wave',fontsize=10,color='#F08080')
    plt.show()
    
    
def cal_two(y1,y2):
    zz = Modified_Z(y1)
    z2 = Modified_Z(y2)
    ind = np.argmax(zz)
    ind2 = np.argmax(z2)
    
    # print(ind)
    # print(ind2)
    if(np.abs(ind - ind2)<100):
        k2 = min(ind,ind2)
        
        n1 = np.append(y1[ind-k2:ind],y1[ind:])
        n2 = np.append(y2[ind2-k2:ind2],y2[ind2:])
        print(len(n1))
        print(len(n2))
        k = cal(n1,n2,len(n1))
    else:
        k = cal(y1,y2,min(len(y1),len(y2)))
    print(k)
    return k




def pearsonrSim(x,y):
    '''
    皮尔森相似度
    '''
    return pearsonr(x,y)[0]
 
 
def spearmanrSim(x,y):
    '''
    斯皮尔曼相似度
    '''
    return spearmanr(x,y)[0]
 
 
def kendalltauSim(x,y):
    '''
    肯德尔相似度
    '''
    return kendalltau(x,y)[0]
 
 
def cosSim(x,y):
    '''
    余弦相似度计算方法
    '''
    tmp=sum(a*b for a,b in zip(x,y))
    non=np.linalg.norm(x)*np.linalg.norm(y)
    return round(tmp/float(non),3)
 
 
def eculidDisSim(x,y):
    '''
    欧几里得相似度计算方法
    '''
    return math.sqrt(sum(pow(a-b,2) for a,b in zip(x,y)))
 
 
def manhattanDisSim(x,y):
    '''
    曼哈顿距离计算方法
    '''
    return sum(abs(a-b) for a,b in zip(x,y))
 
 
def minkowskiDisSim(x,y,p):
    '''
    明可夫斯基距离计算方法
    '''
    sumvalue=sum(pow(abs(a-b),p) for a,b in zip(x,y))
    tmp=1/float(p)
    return round(sumvalue**tmp,3)
 
 
def MahalanobisDisSim(x,y):
    '''
    马氏距离计算方法
    '''
    npvec1,npvec2=np.array(x),np.array(y)
    npvec=np.array([npvec1, npvec2])
    sub=npvec.T[0]-npvec.T[1]
    inv_sub=np.linalg.inv(np.cov(npvec1, npvec2))
    return math.sqrt(np.dot(inv_sub, sub).dot(sub.T))
 
 
def levenshteinDisSim(x,y):
    '''
    字符串编辑距离、相似度计算方法
    '''
    res=Levenshtein.distance(x,y)
    similarity=1-(res/max(len(x), len(y)))
    return similarity
 
 
def jaccardDisSim(x,y):
    '''
    杰卡德相似度计算
    '''
    res=len(set.intersection(*[set(x),set(y)]))
    union_cardinality=len(set.union(*[set(x),set(y)]))
    return res/float(union_cardinality)


    
def plot4():
    s1 = timeit.default_timer()  
    # k = cal(y1,y2,y1.shape[0])

    plt.figure(figsize=(15,8))
    
    plt.subplot(4,1,1)
    plt.plot(range(len(n1)),n1,linewidth=1)
    # plt.scatter(ind2,x2[ind2],marker='p',c='',edgecolors='r',zorder=10)
    plt.ylabel("y1")

    plt.subplot(4,1,2)
    plt.plot(range(len(y1)),y1,linewidth=1)
    # plt.scatter(ind2,x2[ind2],marker='p',c='',edgecolors='r',zorder=10)
    plt.ylabel("z of y1")
    
    plt.subplot(4,1,3)
    plt.plot(range(len(n2)),n2,linewidth=1)
    # plt.scatter(ind2,x2[ind2],marker='p',c='',edgecolors='r',zorder=10)
    plt.ylabel("y2")

    plt.subplot(4,1,4)
    plt.plot(range(len(y2)),y2,linewidth=1)
    # plt.scatter(ind2,x2[ind2],marker='p',c='',edgecolors='r',zorder=10)
    plt.ylabel("z of y2")    
    
    
    plt.show()
    
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))
    
    
def choose(X):

    nums = []
    for i in range(0,X.shape[0]):
        
        temp = X[i,:]
        # print(np.max(temp))
        nums.append(np.max(temp)-np.min(temp))
    print(nums)
    # nums.sort()
    # print(nums)
    max_num_index_list = map(nums.index, heapq.nlargest(1, nums))

    ll = list(max_num_index_list)
    print("Largest dif ref:",ll[0])

def cal(x,y):
    
    print ('pearsonrSim:',pearsonrSim(x,y))
    print ('spearmanrSim:',spearmanrSim(x,y))
    print ('kendalltauSim:',kendalltauSim(x,y))
    print ('cosSim:',cosSim(x,y))
    # print ('eculidDisSim:',eculidDisSim(x,y))
    # print ('manhattanDisSim:',manhattanDisSim(x,y))
    # print ('minkowskiDisSim:',minkowskiDisSim(x,y,2))
    # print ('MahalanobisDisSim:',MahalanobisDisSim(x,y))
    # print ('jaccardDisSim:',jaccardDisSim(x,y))

def main():
    s1 = timeit.default_timer() 
    
    cal(y1,y2)
    # print(bb)
    
    
    s2 = timeit.default_timer()  
    print ('Runing time is (mins):',round((s2 -s1)/60,2))    
if __name__ == '__main__':  
    main()
