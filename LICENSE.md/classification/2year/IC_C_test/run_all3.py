import os
import timeit
# parameter 
import subprocess
# 1st: iteration times of Neat
# 2rd: iteration times of how many runs
# 3nd: lag numbers only 

s1 = timeit.default_timer()  

ff = 3
for i in range(12,25+1):
    out = subprocess.call('python ./ex_oneday16.py '+str(ff)+' 2 '+str(i), shell=True)



for i in range(8,21+1):
    out = subprocess.call('python ./ex_oneday16.py '+str(ff)+' 4 '+str(i), shell=True)


for i in range(3,16+1):
    out = subprocess.call('python ./ex_oneday16.py '+str(ff)+' 6 '+str(i), shell=True)


# for i in range(15,31+1):
    # out = subprocess.call('python ./ex_oneday17.py '+str(ff)+' 12 '+str(i), shell=True)
s2 = timeit.default_timer()  

print ('Runing time is Hour:',round((s2 -s1)/3600,2))