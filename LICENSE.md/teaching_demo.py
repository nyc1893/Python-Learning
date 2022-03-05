# Collection of the code
def fun1():
    print("Hello World")
    print("Hello World2")
    
# Mapping from input to output
def fun2(a):
    return 2*a
    
# Mapping with more input
def fun3(a,b):
    return a+b
    
# 
def BMI(name,height,weight):
    bmi = weight/(height**2)
    print("bmi: " +str(bmi))
    if(bmi>25):
        return name+" is overweight"
    else:
        return name+" is not overweight"    
    
    
name1 = "YC"
h1= 1.83
w1= 82

name2 = "YC's sister"
h2= 1.68
w2= 55   

name3 = "YC's brother"
h3= 1.7
w3= 90 
    
print(BMI(name1,h1,w1))
print(BMI(name2,h2,w2))
print(BMI(name3,h3,w3))


def covert(mile):
    return 1.6*mile  

mile = 120
print(str(mile) +" miles is equal to " +str(covert(mile)) )




##########################################
def iter_fib(k):

    counter = 0
    first = 0
    second = 1
    temp = 0
    
    while counter <= k:
        temp = first + second
        first = second
        second = temp
        counter = counter + 1
    return temp

import time

start = time.time()

iter_fib(20)

end = time.time()

print(end - start)



##########################################


def recur_fibo(k):
    if k <= 1:
        return k
    else:
        return(recur_fibo(k-1) + recur_fibo(k-2))

import time

start = time.time()

recur_fibo(20)

end = time.time()

print(end - start)



