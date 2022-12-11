# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 11:26:31 2022

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 11:14:59 2022

@author: HP
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
import random
from tqdm import tqdm
 
np.seterr(over='ignore')


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



print("/n printing X_Train datasets named as labels /n")
print(X)
#do concatenate X and Y of all 5 batches here
#("/n printing Y_train datasets named as labels/n")
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

def part2(ypred,gt):
   a1=np.shape(ypred)
   b1=np.shape(gt)
   f=0
   for i in range(0,b1[0]):
       if ypred[i] == gt[i]:
           f=f+1
   print("same matched numbers are "+ str(f))        
   print("Efficiency of true_value and Random_Y_Value is "+ str((f/a1[0])*100))


def cifar10_classifier_1nn(x_test,X_trdata,Y_trlabels, Y_test):
    x_test_shape=np.shape(x_test)
    print("The shape of X_test_dataset is "+ str(x_test_shape)+" which has "+ str(x_test_shape[0]) + " rows and "+str(x_test_shape[1])+" columns ")
    
    x_train_shape=np.shape(X_trdata)
    print("The shape of X_train_dataset is "+ str(x_train_shape)+" which has "+ str(x_train_shape[0]) + " rows and "+str(x_train_shape[1])+" columns ")
    
    
    y_train_shape=np.shape(Y_trlabels)
    print("The shape of Y_train_dataset is "+ str(y_train_shape)+" which has "+ str(y_train_shape[0]) + " rows and 1 columns ")
    temp_arr=[] 
    temp_arr2=[]
    sum0=0
    sum1=0
    #abs1=0
    y_pred=[]
    leng=x_train_shape[1]
    for i in tqdm(range(0, 10000)):
       print("printing "+ str(i) +" row of x_test from 10000") 
       for j in range(0, 50000):
           print("printing "+ str(i)+ " row from 10000 and "+ str(j)+" row of X_train from 50000")
           for k in range(0, 3072):
               print("printing "+ str(k)+" column of X_test and X_tr from 3072 and "+str(i)+" row of x_test from 10000 and " + str(j)+ " row of X_train from 50000")
               sum0=x_test[i][k]-X_trdata[j][k]
               sum0=np.abs(sum0)
               temp_arr.append(sum0)
               sum1=np.sum(temp_arr)
           temp_arr=[]
           
           temp_arr2.append(sum1)
           print("list containing all abs sum is "+ str(temp_arr2))
       minindex=temp_arr2.argmin()
       print("Min index is "+str(minindex))
       y_pred.append(Y_trlabels[minindex])
       print("y_pred value at min index of Y_trlabels is "+str(y_pred))
           
    print("The predicted_labels y_pred is "+ str(y_pred))
    
    print("now calling accuracy func.")
    part2(y_pred, Y_test)
    
                    
#arr1.append(min_dist)
print("cifar_10_func now called")
cifar10_classifier_1nn(X_test,X,Y,Y_test)














