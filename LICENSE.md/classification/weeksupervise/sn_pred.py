# from sklearn.datasets import make_moons
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle as pickle  
import pandas as pd
import numpy as np


tr = pd.read_csv('Ltrain_12.csv')
test = pd.read_csv('Ltest_34.csv')


y_tr = tr.pop('real').values
y_test = test.pop('real').values
# test.pop('DT')
tr = tr.values
test = test.values

print(tr.shape)
print(test.shape)


from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


arr = []
for i in range(0,test.shape[0]):
    temp = test[i]
    arr.append(np.argmax(np.bincount(temp)))

arr = np.array(arr)
# print('arrsize:',arr.shape)
from snorkel.labeling import LabelModel

label_model = LabelModel(cardinality= 6, verbose=True)
label_model.fit(L_train=tr, n_epochs=1500, log_freq=100, seed=123)


pred = label_model.predict(test)

label_model_acc = label_model.score(L=test, Y=y_test, tie_break_policy="random")[
    "accuracy"]
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")
test = pd.read_csv('Ltest_34.csv')
test['SNO'] = pred
test['Maj'] = arr
print(test.head())

test.to_csv('Ltest_34.csv',index = None)
print('Ltest_34 update done')
