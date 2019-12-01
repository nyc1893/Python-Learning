#refer
#https://stackoverflow.com/questions/58332191/modulenotfounderror-no-module-named-sklearn-experimental

# 1. upgrade the pip
# 2. update scikit-learn

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)


c = [1,3,4,np.nan,7]
b = [2,6,8,3,np.nan]
aa = {"c":c,"b":b}
a = pd.DataFrame(aa)
# a = [(1, 2), (3, 6), (4, 8), (np.nan, 3), (7, np.nan)]
imp.fit(a)  
# print(type(a))
print(a)

X_test = [(np.nan, 2), (6, np.nan), (np.nan, 6)]
# the model learns that the second feature is double the first
d = pd.DataFrame(imp.transform(X_test))
print(d)
