import os
import timeit
import wk4
import wk5
import wk6
import timeit
import pandas as pd
import numpy as np
# parameter 
# import subprocess
# 1st: iteration times of Neat
# 2rd: iteration times of how many runs
# 3nd: lag numbers only 
# python run_all.py 2>&1 | tee bb.log
s1 = timeit.default_timer()  


num =20
ll = [0.999]
# ll=[0.005,0.01,0.015,0.02]
for j in ll:
    st1= []
    st2= []
    for i in range(num):
        wk4.run(j)
        st1.append(wk5.run())
        tt = wk6.run()
        st2.append(tt)
        # print(tt)

    a1= []
    a2= []
    a3= []
    a4= []

    b1= []
    b2= []
    b3= []
    b4= []

    d1= []
    d2= []
    d3= []
    d4= []

    e1= []
    e2= []
    e3= []
    e4= []

    f1= []
    f2= []
    f3= []
    f4= []

    g1= []
    g2= []
    g3= []
    g4= []

    h1= []
    h2= []
    h3= []
    h4= []
    
    
    c1 =[]
    c2 =[]
    c3 =[]
    c4 =[]
    c5= []
    c6 =[]
    c7 =[]
    for i in range(num):
    
        c1.append(st1[i][0])
        c2.append(st1[i][1])
        c3.append(st1[i][2])
        c4.append(st1[i][3])
        c5.append(st1[i][4])
        c6.append(st1[i][5])
        c7.append(st1[i][6])
        
        
        a1.append(st2[i][0][0])
        a2.append(st2[i][0][1])
        a3.append(st2[i][0][2])
        a4.append(st2[i][0][3])

        b1.append(st2[i][1][0])
        b2.append(st2[i][1][1])
        b3.append(st2[i][1][2])
        b4.append(st2[i][1][3])   
        
        d1.append(st2[i][2][0])
        d2.append(st2[i][2][1])
        d3.append(st2[i][2][2])
        d4.append(st2[i][2][3])

        e1.append(st2[i][3][0])
        e2.append(st2[i][3][1])
        e3.append(st2[i][3][2])
        e4.append(st2[i][3][3])   
        
        f1.append(st2[i][4][0])
        f2.append(st2[i][4][1])
        f3.append(st2[i][4][2])
        f4.append(st2[i][4][3])

        g1.append(st2[i][5][0])
        g2.append(st2[i][5][1])
        g3.append(st2[i][5][2])
        g4.append(st2[i][5][3])        

        h1.append(st2[i][6][0])
        h2.append(st2[i][6][1])
        h3.append(st2[i][6][2])
        h4.append(st2[i][6][3])
        
    ll = ["SNO","semi-DT","semi-GBDT","semi-SVC","semi-LR","semi-KNN","Maj"]
    print("After "+str(num)+" runs, labeled rate: "+str(j))
    print(ll[0])
    print("ACC",round(np.mean(a1),3))
    print("PRE",round(np.mean(a2),3))
    print("REC",round(np.mean(a3),3))
    print("F1",round(np.mean(a4),3))
    
    print(ll[1])
    print("ACC",round(np.mean(b1),3))
    print("PRE",round(np.mean(b2),3))
    print("REC",round(np.mean(b3),3))
    print("F1",round(np.mean(b4),3))
    
    print(ll[2])
    print("ACC",round(np.mean(d1),3))
    print("PRE",round(np.mean(d2),3))
    print("REC",round(np.mean(d3),3))
    print("F1",round(np.mean(d4),3))   

    print(ll[3])
    print("ACC",round(np.mean(e1),3))
    print("PRE",round(np.mean(e2),3))
    print("REC",round(np.mean(e3),3))
    print("F1",round(np.mean(e4),3))
    
    print(ll[4])
    print("ACC",round(np.mean(f1),3))
    print("PRE",round(np.mean(f2),3))
    print("REC",round(np.mean(f3),3))
    print("F1",round(np.mean(f4),3))
    
    print(ll[5])
    print("ACC",round(np.mean(g1),3))
    print("PRE",round(np.mean(g2),3))
    print("REC",round(np.mean(g3),3))
    print("F1",round(np.mean(g4),3))   

    print(ll[6])
    print("ACC",round(np.mean(h1),3))
    print("PRE",round(np.mean(h2),3))
    print("REC",round(np.mean(h3),3))
    print("F1",round(np.mean(h4),3))   
    
    
    df = pd.DataFrame(a1)
    k = 0
    df.columns = [ll[k]+'-ACC']
    df[ll[k]+'-PRE'] = a2
    df[ll[k]+'-REC'] = a3
    df[ll[k]+'-F1'] = a4
    
    k = 1
    df[ll[k]+'-ACC'] = b1
    df[ll[k]+'-PRE'] = b2
    df[ll[k]+'-REC'] = b3
    df[ll[k]+'-F1']  = b4
    
    k = 2
    df[ll[k]+'-ACC'] = d1
    df[ll[k]+'-PRE'] = d2
    df[ll[k]+'-REC'] = d3
    df[ll[k]+'-F1']  = d4
    
    k = 3
    df[ll[k]+'-ACC'] = e1
    df[ll[k]+'-PRE'] = e2
    df[ll[k]+'-REC'] = e3
    df[ll[k]+'-F1']  = e4
    
    k = 4
    df[ll[k]+'-ACC'] = f1
    df[ll[k]+'-PRE'] = f2
    df[ll[k]+'-REC'] = f3
    df[ll[k]+'-F1']  = f4
        
    k = 5
    df[ll[k]+'-ACC'] = g1
    df[ll[k]+'-PRE'] = g2
    df[ll[k]+'-REC'] = g3
    df[ll[k]+'-F1']  = g4
    
    k = 6
    df[ll[k]+'-ACC'] = h1
    df[ll[k]+'-PRE'] = h2
    df[ll[k]+'-REC'] = h3
    df[ll[k]+'-F1']  = h4
    
    df.to_csv("res.csv",index = 0)
    print("save done")
    print("label---------------")
    print(ll[0] +":"+ str(round(np.mean(c1),3)))
    print(ll[1] +":"+ str(round(np.mean(c2),3)))
    print(ll[2] +":"+ str(round(np.mean(c3),3)))
    print(ll[3] +":"+ str(round(np.mean(c4),3)))
    print(ll[4] +":"+ str(round(np.mean(c5),3)))
    print(ll[5] +":"+ str(round(np.mean(c6),3)))
    print(ll[6] +":"+ str(round(np.mean(c7),3)))
    
    df = pd.DataFrame(c1)
    k=0
    df.columns = [ll[k]]
    k=1
    df[ll[k]] = c2
    k=2
    df[ll[k]] = c3
    k=3
    df[ll[k]] = c4
    k=4
    df[ll[k]] = c5
    k=5
    df[ll[k]] = c6
    k=6
    df[ll[k]] = c7
    
    df.to_csv("label.csv",index = 0)
    
s2 = timeit.default_timer()  


print ('Runing time is mins:',round((s2 -s1)/60,2))

    # out = subprocess.call('python ./wk1-3.py', shell=True)

    # out = subprocess.call('python ./wk2-2.py', shell=True)
    
    # out = subprocess.call('python ./wk3-2.py', shell=True)
# print(st1)
# print(st2)
# print("qiepian")
# print("run 1:",st2[0])
# print("SNO :",st2[0][0])



# print("SNO ACC:",st2[0][0][0])
# print("SNO PRE:",st2[0][0][1])
# print("SNO Rec:",st2[0][0][2])
# print("SNO F1:",st2[0][0][3])

# print("Semi ACC:",st2[0][1][0])
# print("Semi PRE:",st2[0][1][1])
# print("Semi Rec:",st2[0][1][2])
# print("Semi F1:",st2[0][1][3])
