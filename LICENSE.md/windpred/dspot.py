import math
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

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
	# print ("Inital Threshold", t)
	# print ("Updated Threshold", zq)
	# print ("len nt = ", len(nt))
	# print ("len yt = ", len(yt))
	
data = pd.read_csv("../data/mit_2009.csv")
# input 
# n: lens of calibration data
# d: window size
# q: quantile


i = 0 #initial point
d = 500
n =1500
df = data.loc[0:d]
x = data.values
# x = np.arange(1,52060+1)

# print(df.shape)
# df2 = data.loc[d+1:d+n]
# print(df2.shape)
# df3 = data.loc[d+n+1:]
# print(df3.shape)
n2 = len(data.values)
q = 95

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
print("zq",zq)
print("t",t)

zzq =zq*np.ones(n2)
# A is a set of anomalies
Avalue = []
Aindex = []

k = n
k2 = 52060-n-d
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


print("Anomaly case number = ",len(Avalue))
# plt.plot(line,zzq,color='green')
plt.plot(line,x,color='blue')
# plt.plot(line,xp,color='blue')
	
		
xais = np.arange(1,len(Avalue)+1)		
plt.plot(Aindex,x[Aindex],color='red')		
plt.show()			
		
		
