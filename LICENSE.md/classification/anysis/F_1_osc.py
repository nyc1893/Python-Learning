import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

N = 600
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)


y = [0,2,3,4,2]*4
y = np.array(y)
print('y',y.shape)
yf = fft(y)
print('yf',yf.shape)

def get_B(y):
    res = [] 
    for i in range(len(y)-1):
        if(y[i-1]<y[i]):
            res.append(-1)
        else:
            res.append(1)
    res = np.array(res)
    return res
    
def get_C(b):
    c1 = 1
    c2 = 1
    # temp = b[0]
    res =[]
    for i in range(1,len(b)):
        if(b[i]== b[i-1]):
            if(b[i] ==1):
                c1+=1
            elif(b[i] ==-1):
                c2+=1
        else:

            if(b[i] ==1):
                res.append(c2)
            elif(b[i] ==-1):
                res.append(c1)          

            c1 = 1
            c2 = 1    
    return res
    
    
def get_D(b):
    res =0
    for i in range(0,len(b)-2):
        if(np.abs(b[i]-b[i+2])<1):
            res+=b[i]
    if(np.abs(b[len(b)-1]-b[len(b)-3])<1):
        res+=b[len(b)-1]
        
    if(np.abs(b[len(b)-2]-b[len(b)-4])<1):
        res+=b[len(b)-2]        
    return res    
    
    
def pp(y):  
    num = 8
    n=y.shape[0]# 信号长度
    t = range(n)
    yy=fft(y)
    yf=abs(yy)#取绝对值
    yf1=abs(fft(y))/n#归一化处理
    yf2=yf1[range(int(n/num))]##由于对称性，只取一半区间

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
    cc = get_B(y)
    dd = get_C(cc)
    ee = get_D(dd)
    plt.subplot(212)
    plt.plot(range(n-1),cc,'g')
    plt.xlabel(str(dd)+"; "+str(ee))
    plt.show()
    
pp(y)
