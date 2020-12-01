#This one will get rid of Prefix like P1_...
#!/usr/bin/env python
# coding=utf-8
import os
#p1 is the folder name tat contain all the file
name = 'p10'
dname = '/team_mgmt'

path = "./"+name+"/"
path2 = "./" + name
fileList=os.listdir(path2)
for f in fileList:
    # x = f.split("-", 1)
    ff = os.listdir(str(path)+f)
    for f2 in ff:
        if "_mgmt"  in f2:
        # if(f2.str.contain('_mgmt')):
            print(str(path)+f)
            os.remove(str(path)+f+dname)


