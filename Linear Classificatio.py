# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 01:14:48 2022

@author: HP
"""

#Question #2:
    
import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

datadict1 = unpickle(r'E:\Profession\Finland\Studies\SemOne\ML100IntrotoPatternRecogandML\weekThree\cifar-10-batches-py\data_batch_1')
#datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/test_batch')
datadict2 = unpickle(r'E:\Profession\Finland\Studies\SemOne\ML100IntrotoPatternRecogandML\weekThree\cifar-10-batches-py\data_batch_2')
datadict3 = unpickle(r'E:\Profession\Finland\Studies\SemOne\ML100IntrotoPatternRecogandML\weekThree\cifar-10-batches-py\data_batch_3')
datadict4 = unpickle(r'E:\Profession\Finland\Studies\SemOne\ML100IntrotoPatternRecogandML\weekThree\cifar-10-batches-py\data_batch_4')
datadict5 = unpickle(r'E:\Profession\Finland\Studies\SemOne\ML100IntrotoPatternRecogandML\weekThree\cifar-10-batches-py\data_batch_5')

print("this is datadict/n")
#print(datadict)

#X_Train and Y_train data for batch_1
X1 = np.array(datadict1["data"]) #X_train_for batch_1
print("size of X1 batch is")
print(np.shape(X1))
print(X1)

Y1 = np.array(datadict1["labels"]) #Y_Train_for-batch_1
print("size of Y1 batch is")
print(np.shape(Y1))
print(Y1)

#X_Train and Y_train data for batch_2
X2 = np.array(datadict2["data"]) #X_train_for batch_2
print("size of X2 batch is")
print(np.shape(X2))
print(X2)

Y2 = np.array(datadict2["labels"]) #Y_Train_for-batch_2
print("size of Y2 batch is")
print(np.shape(Y2))
print(Y2)



#X_Train and Y_train data for batch_3
X3 = np.array(datadict3["data"]) #X_train_for batch_3
print("size of X3 batch is")
print(np.shape(X3))


Y3 = np.array(datadict3["labels"]) #Y_Train_for-batch_3
print("size of Y3 batch is")
print(np.shape(Y3))




#X_Train and Y_train data for batch_4
X4= np.array(datadict4["data"]) #X_train_for batch_4
print("size of X4 batch is")
print(np.shape(X4))



Y4 = np.array(datadict4["labels"]) #Y_Train_for-batch_4
print("size of Y4 batch is")
print(np.shape(Y4))



#X_Train and Y_train data for batch_5
X5 = np.array(datadict5["data"]) #X_train_for batch_5
print("size of X5 batch is")
print(np.shape(X5))

Y5 = np.array(datadict5["labels"]) #Y_Train_for-batch_5
print("size of Y5 batch is")
print(np.shape(Y5))


#now_concatinate_whole_5_batches_into_1_array_of_X_train, Y_train

X=np.concatenate((X1,X2,X3,X4,X5),axis=0)

Y=np.concatenate((Y1,Y2, Y3,Y4,Y5),axis=0)

#X=np.concatenate((Xc,X4,X5),axis=0)

#Y=np.concatenate((Yc,Y4,Y5),axis=0)




print("/n printing X data set named as labels /n")
print(X)
#do concatenate X and Y of all 5 batches here
("/n printing Y data sets named as labels/n")
print("now Y will print")
print(Y)

print("SIZE OF X_Train")
print(np.shape(X))

print("SIZE OF Y_Train")
print(np.shape(Y))
a=np.shape(Y)
print(a[0])



labeldict = unpickle(r'E:\Profession\Finland\Studies\SemOne\ML100IntrotoPatternRecogandML\weekThree\cifar-10-batches-py\batches.meta')
label_names = labeldict["label_names"]

print("calling test batch data now")
datadict_test = unpickle(r'E:\Profession\Finland\Studies\SemOne\ML100IntrotoPatternRecogandML\weekThree\cifar-10-batches-py\test_batch')


X_test = np.array(datadict_test["data"])
Y_test = np.array(datadict_test["labels"])

print("shape of X_test dataset is "+ str(np.shape(X_test)))
print("shape of Y_test dataset is "+str(np.shape(Y_test)))


def class_acc(ypred,gt):
    a1=np.shape(ypred)
    b1=np.shape(gt)
    f=0
    for i in range(0,b1[0]):
        if ypred[i] == gt[i]:
            f=f+1
    print("same matched numbers are "+ str(f))        
    print("Accuracy is "+ str((f/a1[0])*100) + "%")
    
    
    
class_acc(Y_test,Y_test)
     
    
    
X = X.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Y)


for i in range(X.shape[0]):
    # Show some images randomly
    if random() > 0.999:
        plt.figure(1);
        plt.clf()
        plt.imshow(X[i])
        plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
        plt.pause(1)

#X=X.reshape(10000,(3072))






