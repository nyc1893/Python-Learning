import os
import timeit
# parameter 
import subprocess
# 1st: iteration times of Neat
# 2rd: iteration times of how many runs
# 3nd: lag numbers only 

s1 = timeit.default_timer()  

list = range(16,32) 
for i in range(1,len(list)):
    out = subprocess.call('python ./dy2.py 11 '+str(list[i-1])+' '+ str(list[i]), shell=True)
s2 = timeit.default_timer()  

list = range(1,14) 
for i in range(1,len(list)):
    out = subprocess.call('python ./dy2.py 12 '+str(list[i-1])+' '+ str(list[i]), shell=True)
s2 = timeit.default_timer()  


print ('Runing time is Hour:',round((s2 -s1)/3600,2))
