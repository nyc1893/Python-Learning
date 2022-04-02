from rpy2.robjects.packages import importr
# import R's "base" package
base = importr('base')
# import R's "utils" package
utils = importr('utils')


import rpy2.robjects as robjects

res = robjects.StrVector(['abc', 'def'])
print(res.r_repr())
res = robjects.IntVector([1, 2, 3])
print(res.r_repr())
res = robjects.FloatVector([1.1, 2.2, 3.3])
print(res.r_repr())

robjects.r('v=c(1.1, 2.2, 3.3, 4.4, 5.5, 6.6)') #在R空间中生成向量v
m = robjects.r('matrix(v, nrow = 2)') #在R空间中生成matrix，并返回给python中的对象m
print(m)
