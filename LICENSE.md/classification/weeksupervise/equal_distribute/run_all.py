import os
import timeit
# parameter 
import subprocess
# 1st: iteration times of Neat
# 2rd: iteration times of how many runs
# 3nd: lag numbers only 
# python run_all.py 2>&1 | tee bb.log
s1 = timeit.default_timer()  

for i in range(1):
    out = subprocess.call('python ./wk1-3.py', shell=True)

    out = subprocess.call('python ./wk2-2.py', shell=True)
    
    out = subprocess.call('python ./wk3-2.py', shell=True)
    
s2 = timeit.default_timer()  


print ('Runing time is mins:',round((s2 -s1)/60,2))