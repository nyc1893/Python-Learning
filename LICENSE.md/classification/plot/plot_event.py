# from sklearn.datasets import make_moons
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle as pickle  
import pandas as pd
import numpy as np

def removePlanned(X,y):
    """
    THIS FUNCTION REMOVES THE PLANNED EVENTS FROM THE EVENT DATASET
    """
    
    X_new=[]
    y_new=[]
    for i in range(len(y)):
        #print(i)
    
        if y[i]==0:
            y_new.append(0)
            X_new.append(X[i,:,:,:])
    
        elif y[i]==1:
            y_new.append(1)
            X_new.append(X[i,:,:,:])
    
            
        elif y[i]==2:
            y_new.append(2)
            X_new.append(X[i,:,:,:])
        
        elif y[i]==3:
            y_new.append(3)
            X_new.append(X[i,:,:,:])
        

    return  np.array(X_new), np.array(y_new)


def separatePMUs(X,y):
    
    """
    This function separates features and their corresponding labels for each PMU
    to make more events 
    """
    
    num_case=X.shape[0]
    num_pmu=X.shape[1]
    num_sample=X.shape[2]
    X=X.reshape(num_case*num_pmu,num_sample)
    y_new=[]
    for i in range(len(y)):
        if y[i]==0:
            for j in range(num_pmu):
                y_new.append(0)
                
        if y[i]==1:
            for j in range(num_pmu):
                y_new.append(1)
                
        if y[i]==2:
            for j in range(num_pmu):
                y_new.append(2)
                
        if y[i]==3:
            for j in range(num_pmu):
                y_new.append(3)
        
    
    return X,np.array(y_new)

def read_data():  
    # path2 = '../svm/'
    # df1 = pd.read_csv(path2 +'X_train.csv')
    # df2 = pd.read_csv(path2 +'X_test.csv')
    # df1 = pd.concat([df1,df2])
    
    path = '../pickleset/'
    p2 = open(path+ "X_S2_rocof_6.pickle","rb")
    pk2 = pickle.load(p2)
    X_test = pk2
    p2 = open(path+ "y_S2_rocof_6.pickle","rb")
    pk2 = pickle.load(p2)
    y_test = pk2
    
    # y_train = df1.pop('label')
    # X_train = df1
    
    
    fps=60
    start_crop=int(fps*60*4)
    stop_crop=int(fps*60*8)

    X_test= X_test[:,:,start_crop:stop_crop,:]
    X_test, y_test=removePlanned(X_test, y_test)
    X_test, y_test= separatePMUs(X_test,y_test)
    # X_test, y_test = shuffle(X_test,y_test)    
    
    # print('X_train.shape',X_train.shape)
    # print('y_train.shape',y_train.shape)
    print('X_test.shape',X_test.shape)
    print('y_test.shape',y_test.shape)    
    
    return X_test,y_test  
    # return X_train,y_train,X_test,y_test   
    
def plot_eve():
    X_test,y_test  = read_data()
    
    
    
def main():
    
    plot_eve()


if __name__ == '__main__':  
    main()

