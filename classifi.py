
import csv

import numpy as np

with open('HANUMAN.csv',encoding='utf-8-sig') as f:
    point_history_classifier_labels = csv.reader(f)
    labels1 = [row[0] for row in point_history_classifier_labels]


 
dataset=np.loadtxt('HANUMAN.csv', delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))

from sklearn.model_selection import train_test_split
vb=np.array(labels1)
labels1=vb.astype(np.uint8)

Y=labels1
X=dataset    


from tensorflow.keras.utils import to_categorical
 
# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.2, 
                                                    random_state=12)

# Print number of observations in X_train, X_test, y_train, and y_test
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

y_train = to_categorical( np.array(y_train), max(Y)+1) 

y_test = to_categorical( np.array(y_test), max(Y)+1) 

#######

# graph the history of model.fit
def show_history_graph(history):
    # summarize history for accuracy
    plt.plot(np.array((hist.history['accuracy'])))
    plt.plot(np.array((hist.history['val_accuracy'])))
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# define the keras model
model = Sequential()
model.add(Dense(128, input_dim=42, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(max(Y)+1, activation='softmax'))
# compile the keras model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset


hist=model.fit(X_train, y_train ,validation_split=0.1, epochs=100, batch_size=32)
#hist=model.fit(X_train, y_train, epochs=15, batch_size=10)
show_history_graph(hist)
test_loss, test_acc = model.evaluate(X_test, y_test)

y_pred=model.predict(X_train)
y_test1=y_test
y_test=y_train
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(np.argmax(y_test, axis=1),np.argmax(y_pred, axis=1))

accuracy = accuracy_score(np.argmax(y_test, axis=1),np.argmax(y_pred, axis=1))
print("CNN confusion matrics=",cm)
print("  ")
print("CNN accuracy=",(accuracy)*100)
nn=(accuracy)*100

from sklearn.metrics import roc_curve
# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1)>=1,np.argmax(y_pred, axis=1)>=1)



# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('CNN True Positive Rate')
plt.xlabel('CNN False Positive Rate')
plt.show()

model.save('trained_model_CNN.h5')
 


