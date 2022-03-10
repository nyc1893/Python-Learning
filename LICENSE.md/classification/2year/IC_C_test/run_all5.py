import os
import timeit
# parameter 
import subprocess
# 1st: iteration times of Neat
# 2rd: iteration times of how many runs
# 3nd: lag numbers only 

s1 = timeit.default_timer()  

ff = 3
for i in range(29,31+1):
    out = subprocess.call('python ./ex_oneday16.py '+str(ff)+' 7 '+str(i), shell=True)



for i in range(1,11+1):
    out = subprocess.call('python ./ex_oneday16.py '+str(ff)+' 8 '+str(i), shell=True)


for i in range(23,30+1):
    out = subprocess.call('python ./ex_oneday16.py '+str(ff)+' 9 '+str(i), shell=True)


for i in range(1,6+1):
    out = subprocess.call('python ./ex_oneday16.py '+str(ff)+' 10 '+str(i), shell=True)
s2 = timeit.default_timer()  

print ('Runing time is Hour:',round((s2 -s1)/3600,2))