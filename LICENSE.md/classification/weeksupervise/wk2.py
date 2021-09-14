# from sklearn.datasets import make_moons
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle as pickle  
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=100)  
    model.fit(train_x, train_y)  
    return model  
  

# python wk2.py 2>&1 | tee sno.log
def get_sno_label():
    tr = pd.read_csv('data/wk_1_tr.csv')
    test = pd.read_csv('data/wk_1_test.csv')
    tr.pop("LR")
    test.pop("LR")
    y_tr = tr.pop('real').values
    y_test = test.pop('real').values
    # test.pop('RF')
    tr = tr.values
    test = test.values

    print(tr.shape)
    print(test.shape)




    arr = []
    for i in range(0,test.shape[0]):
        temp = test[i]
        arr.append(np.argmax(np.bincount(temp)))

    arr = np.array(arr)
    # print('arrsize:',arr.shape)
    from snorkel.labeling import LabelModel

    label_model = LabelModel(cardinality= 3, verbose=True)
    label_model.fit(L_train=tr, n_epochs=1500, log_freq=100, seed=123)


    pred = label_model.predict(test)

    label_model_acc = label_model.score(L=test, Y=y_test, tie_break_policy="random")[
        "accuracy"]
    print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")
    test = pd.read_csv('data/wk_1_test.csv')
    test.pop('LR')
    jj = test.pop('real')
    # print(test.head())
    model = random_forest_classifier(tr, y_tr) 
    rf_pred = model.predict(test)  
    
    test['SNO'] = pred
    test['Maj'] = arr
    test['rf_pred'] = rf_pred
    test['real'] = jj
    test.to_csv('wk_2_test.csv',index = None)

def esti():
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    df = pd.read_csv('wk_2_test.csv')
    y_test = df["real"]

    ll = df.columns.values
    # ll =["RF","GBDT"]
    for i in range(0,df.shape[1]-1):
        
        y_pred = df[ll[i]]
        print('\n testing error of ' + str(ll[i])) 
        print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

        # print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
        # print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
        # print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

        print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
        print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
        print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

        # print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
        # print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
        # print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))    
        
        
        
        matrix=confusion_matrix(y_test, y_pred)
        print(matrix)
        class_report=classification_report(y_test, y_pred)
        print(class_report)
        

get_sno_label()
esti()
# print('wk_2_test update done')



