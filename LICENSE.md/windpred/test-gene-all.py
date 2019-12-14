# This is for after normal and extreme prediction
# how to index it in the right order so that can combine GE and mits data together
# for further research

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from test_neat import test_neat

"""
data = {'name':['a','b','c','d','e'],
	   'number':[1,2,3,4,5]
	   }
df = pd.DataFrame(data)
print(df)
idx = df.index
print(idx.tolist())

list1 = [0,2,4]
list2 = [1,3]


d1 = df.iloc[[0, 2,4],:]
d2 = df.iloc[[1,3],:]

print(d1)
print(d2)

df2 = pd.concat([d1, d2], axis=0)
# print(df2)

# print(df2)

df3 = df2.sort_index(inplace=False)
print(df3)

print(df2)
"""
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
	
	zq = t+ (sig/gam)*(((q*n/no_t)**(-gam))-1)	 #function(1)
	return zq,t

# This will get the result of running neats
from run_neat import run_neat
from neat_pred import neat_pred


# input 
# n: lens of calibration data
# d: window size
# q: quantile
def fun1(q,d,n,show,year,turb):

	i = 0 #initial point
	df1 = pd.read_csv("../data/total_"+str(turb)+"_"+str(year)+".csv")	  
	data = pd.read_csv("../data/total_"+str(turb)+"_"+str(year+1)+".csv")

	df1[df1<0] = 0
	data[data<0] = 0
	# print(df1.shape)
	df1 = df1.iloc[-(d+n):]
	# print(df1.shape)
	data = pd.concat([df1, data], axis=0)
	data=data.reset_index(drop = True)
	# print(df1.head())
	# print(data.head())

	# print('data.shape',data.shape)

	df = data.loc[0:d]
	x = data.values

	print("lens of data: ",len(x))


	# x = np.arange(1,52060+1)

	# figsize(16, 4)
	# print(df.shape)
	# df2 = data.loc[d+1:d+n]
	# print(df2.shape)
	# df3 = data.loc[d+n+1:]
	# print(df3.shape)
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
	data.loc[Aindex,'d'] = 1
	# print("Anomaly case number = ",len(Avalue))
	# print(data.head(20))
	print(data['d'].mean())
	data =	data.iloc[(n+d):]
	# print(data.shape)
	label1 = data['d']
	# idx = label1.index.tolist()
	# idx[:] = [x - (1) for x in idx]
	# label1 = label1.iloc[idx,:]
	label1 = label1.reset_index(drop=True)
	# label1.to_csv("labelaaa.csv")
	return label1

	

	
	# print(pred.head())
	# print(pred.shape)
	return pred




turb = 'mit'
q1= 70
q2= 99
d1 = 717
d2 = 498
i = 1
# pred = get_result(q1,q2,d1,d2,turb,i,'a','gene2/winner_mit')

year = 2008
n =1500



df1 = fun1(q1,d1,n,0,year,turb) 
df2 = fun1(q2,d2,n,0,year+1,turb)  
# print("df1.mean()",df1.mean())
# print(df1.head())
# print(df1.tail())
# idx = df1.index.tolist()
# idx[:] = [x - (2210) for x in idx]

# df = df1.iloc[idx,:]
print(df2.head())
print(df2.tail())

test  = pd.read_csv('../data/pp'+str(turb)+'12_'+str(year+2)+'.csv')


df2 = df2.values.tolist()
print(len(df2))
df2 = df2 [8:]
print(len(df2))
df2 = pd.DataFrame(df2)
df2.columns = ['d']
print(df2.shape)

# df2 = df2.reset_index(drop=True)
print('df2 shape: ',df2.shape)
print('test shape: ',test.shape)



# tr = pd.concat([tr, df1], axis=1,join_axes=[tr.index])
test = pd.concat([test, df2], axis=1,join_axes=[test.index])
# print('tr shape: ',tr.shape)
print('test shape: ',test.shape)
# test = test.iloc[1500+ d2:,:]
# print('tr head: ',tr.head())

df6 = test[test['d']==0]
df4 = test[test['d']==1]
# print(test.head(30))
idx1 = df6.index
print(idx1.tolist()[0:10])
idx2 = df4.index
print(idx2.tolist()[0:10])




df4.pop('d')
df6.pop('d')

print("Before pred extreme",df4.shape)
print("Before pred normal",df6.shape)


dd2 = pd.concat([df6, df4], axis=0)
# print(df2)

# print(df2)

dd3 = dd2.sort_index(inplace=False)
# print(dd3.head(10))

y_test =  df4.pop('l0')
X_test = df4
ddf6 =	df6.pop('l0')
print(df6.shape)
print(ddf6.shape)

print(X_test.shape)
print(y_test.shape)


X_test = X_test.values.tolist()
y_test = y_test.values.tolist()

df6 = df6.values.tolist()
ddf6 = ddf6.values.tolist()


p2 = neat_pred(df6,ddf6,turb)
p1 = test_neat(X_test,y_test,i,'a','gene2/winner_mit')

print("After pred extreme",p1.shape)
print("After pred normal",p2.shape)

# print("nan number3=",np.isnan(p2).sum())

p1 = pd.DataFrame(p1)
pp = pd.DataFrame(idx2.tolist())
# p1 = p1.reindex(idx2)
p1 = pd.concat([pp, p1], axis=1,join_axes=[p1.index])
p1.columns = ['ind','pred1']
# p1.to_csv("p1.csv")


p2 = pd.DataFrame(p2)
pp = pd.DataFrame(idx1.tolist())
p2 = pd.concat([pp, p2], axis=1,join_axes=[p2.index])
p2.columns = ['ind','pred2']
# p2.to_csv("p2.csv")

pred = pd.merge(p1, p2, how='outer', on=['ind'])
pred = pred.fillna(0)
pred['out'] = pred['pred1'] + pred['pred2']
pred = pred.sort_values(by='ind')
# pred.to_csv("p1.csv")
print("nan of pred",np.isnan(pred).sum())
y = pred['out'].values

test = pd.read_csv('../data/pp'+str(turb)+'12_'+str(year+2)+'.csv')
yhat = test.pop('l0') 
max_y = 221
mae= mean_absolute_error(y, yhat)
print("Mits NMAE=",100*mae/max_y)
