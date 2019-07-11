"""
Visualize Genetic Algorithm to find a maximum point in a function.
Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""



from google.colab import files
import numpy as np
import pandas as pd

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

df=pd.read_csv('iris.csv')
print(df.head(20))
data = df
from pandas.api.types import is_numeric_dtype
for col in data.columns:
    if is_numeric_dtype(data[col]):
        print('%s:' % (col))
        print('\t Mean = %.2f' % data[col].mean())
        print('\t Standard deviation = %.2f' % data[col].std())
        print('\t Minimum = %.2f' % data[col].min())
        print('\t Maximum = %.2f' % data[col].max())
        
data['label'].value_counts()        

df1 = df[df['label'] == 'Iris-setosa']
df2 = df[df['label'] == 'Iris-versicolor']
df3 = df[df['label'] == 'Iris-virginica']  

for i in range(0,2+1):
    if i ==0:
        temp = df1
        print('\nFor setosa:')
    elif i ==1:
        temp = df2
        print('\nFor versicolor:')
    else:
        temp = df3
        print('\nFor virginica:')
    for col in temp.columns:
        if is_numeric_dtype(temp[col]):
            print('%s:' % (col))
            print('\t Mean = %.2f' % temp[col].mean())
            print('\t Standard deviation = %.2f' % temp[col].std())
            print('\t Minimum = %.2f' % temp[col].min())
            print('\t Maximum = %.2f' % temp[col].max())

            
          
for i in range(1,3+1): 
    print('\n')
    locals()['df'+str(i)].boxplot()             
    for col in locals()['df'+str(i)].columns:
        if is_numeric_dtype(locals()['df'+str(i)][col]):
            print('%s:' % (col))
            print('\t Mean = %.2f' % locals()['df'+str(i)][col].mean())
            print('\t Standard deviation = %.2f' % locals()['df'+str(i)][col].std())
            print('\t Minimum = %.2f' % locals()['df'+str(i)][col].min())
            print('\t Maximum = %.2f' % locals()['df'+str(i)][col].max())   


label = data.pop('class')
lens = 120
X_train = data.loc[range(0,lens),:]
X_test = data.loc[range(lens,150),:]

y_train = label[0:lens]
y_test = label[lens:150]
print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)

           
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=10)
clf = clf.fit(X_train, y_train)
predY = clf.predict(X_test)
predictions = pd.Series(predY,name='Predicted Class')
print(predictions)

print(y_test)
            
from sklearn.metrics import accuracy_score
print('Accuracy on test data is %.2f' % (accuracy_score(y_test, predY)))            
