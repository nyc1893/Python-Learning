
#https://9xbuddy.org/process?url=https%3A%2F%2Fcn.pornhub.com%2Fview_video.php%3Fviewkey%3Dph5ce8d9ca772ca
import os
# os.system("ffprobe -show_format v1.mp4 ")


import subprocess,json


def get_length(path,mins):
    pname='ffprobe -v quiet -print_format json -show_format '+str(path)
    result=subprocess.Popen(pname,shell=False,stdout=subprocess.PIPE).stdout
    list_std=result.readlines()
    # print(list_std)
    # print('list--------------------')
    str_tmp=''
    for item in list_std:
        str_tmp+=bytes.decode(item.strip())
    json_data=json.loads(str_tmp)
    # print('json_data---------------')
    # print(json_data)
    dura_time = json_data['format']['duration']

     
    # print(dura_time)

    return int(float(dura_time))/(60*mins)


src = 'p1/v1.mp4'
dir = 'p1/'
name = 'ck'


num = 4
# list = ['00:00:00','00:03:00']

# Only the starting point:
list = ['00:00:00','00:00:30','00:01:00','00:01:30']
for i in range(0,num):
    
    start = list[i]
    end = list[1]
    print(start)
    print(end)
    os.system("ffmpeg -i "+str(src)+" -ss "+str(start)+" -t "+str(end)+" -c copy "+str(dir)+str(name)+"-"+str(i)+".mp4")



# ffmpeg -i v1.mp4 -ss 00:00:00 -t 00:01:00 -c copy cc.mp4
