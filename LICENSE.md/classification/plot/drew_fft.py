import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

N = 600
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)


y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)

print('y',y.shape)
yf = fft(y)
print('yf',yf.shape)

def date_fft(y):
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
    plt.plot(xf2, yf2, 'b')
    # plt.bar(xf2, yf2, alpha=0.99, width = 1, facecolor = 'blue', edgecolor = 'white', label='one', lw=1)
    # 注意这里的颜色可以查询颜色代码表
    plt.xlabel("Freq (Hz)")
    plt.ylabel("|Y(freq)|")
    plt.title('FFT of Mixed wave',fontsize=10,color='#F08080')
    plt.show()
    
    
date_fft(y)
