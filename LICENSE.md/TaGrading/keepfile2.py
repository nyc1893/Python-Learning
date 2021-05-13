
# 目录里面有吧文件夹， 删掉除了 "coding-problem-2" 以外的所有文件和文件夹
import os
import os.path
import shutil

# this is 2 layer folder processing
def deal(path2):


    path2 = path2+"/"
    fileList=os.listdir(path2)
    remainDirsList ="coding-problem-2"

    for f in fileList:

        if f not in remainDirsList and os.path.isdir(path2+f):
            print(path2+f)
            shutil.rmtree(path2+f) 

        elif os.path.isfile(path2+f):
            os.remove(path2+f)
            

def top(ii):
    path ="exam"+str(ii)+"/"
    fileList=os.listdir(path)
    for f in fileList:    
        print(path+f)
        deal(path+f)
            
def main():

    top(7)
        
        
main()
