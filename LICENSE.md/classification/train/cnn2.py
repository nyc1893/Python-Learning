import numpy as np
import pandas as pd
import keras
import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.optimizers import Adam

c_num = 4
def recover(df):
    df = pd.DataFrame(df)
    df.columns = ['L1','L2','L3','L4']
    dd = np.zeros(df.shape[0])
    dd = dd.astype("int8") 
    rr = df[df['L2']>0].index.tolist()
    dd[rr] = 1
    
    rr = df[df['L3']>0].index.tolist()
    dd[rr] = 2    


    rr = df[df['L4']>0].index.tolist()
    dd[rr] = 3    
    
    df['rev'] = dd
    return df['rev'].values 
    
    
channel_num = 4
s_x = 80
s_y = 77

x_train = np.random.random((100, s_x, s_y, channel_num))
y_train = keras.utils.to_categorical(np.random.randint(c_num, size=(100, 1)), num_classes=c_num)
# y_train = (np.random.randint(c_num, size=(100, 1)), num_classes=c_num)
x_test = np.random.random((20, s_x, s_y, channel_num))
y_test = keras.utils.to_categorical(np.random.randint(c_num, size=(20, 1)), num_classes=c_num)
# y_test = (np.random.randint(c_num, size=(20, 1)), num_classes=c_num)

model = Sequential()

model.add(Conv2D(32, (channel_num, 3), activation='relu', input_shape=(s_x, s_y, channel_num)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd)

optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=32, epochs=10)

print(y_test.shape)
# print(y_train.shape)

y_test = recover(y_test)




y_pred=model.predict_classes(x_test) 

print(y_pred.shape)
# y_pred = recover(y_pred)


# print(y_test.shape)
print(y_pred.shape)

matrix=confusion_matrix(y_test, y_pred)
print(matrix)
class_report=classification_report(y_test, y_pred)
print(class_report)


