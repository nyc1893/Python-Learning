import os
import timeit

list = range(1,9) 

s1 = timeit.default_timer()  

for i in range(1,len(list)):
    os.system('python ./dy.py 4 '+str(list[i-1])+' '+ str(list[i]))
    
s2 = timeit.default_timer()  
print ('Runing time is Hour:',round((s2 -s1)/3600,2))
