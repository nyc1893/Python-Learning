#This one will get rid of Prefix like P1_...
#!/usr/bin/env python
# coding=utf-8
import os
#p1 is the folder name tat contain all the file
name = 'p10'
path = "./"+name+"/"
path2 = "./" + name
fileList=os.listdir(path2)
for f in fileList:
    x = f.split("-", 1)
    
    
    
    srcDir = str(path)+f
    dstDir = str(path)+x[1]
    # print(f)
    # print(x[1])
    try:
        os.rename(srcDir,dstDir)
    except Exception as e:
        print (e)
        print ('rename dir fail\r\n')
    else:
        print ('rename dir success\r\n')
