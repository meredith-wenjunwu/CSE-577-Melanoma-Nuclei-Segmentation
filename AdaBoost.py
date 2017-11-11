#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 6 18:26:58 2017

@author: shima
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

#### Functions

def Adaboost(X_train , Y_train , X_test, Y_test, iterations, classifier):

    # Initialize
    w = np.ones(X_train.shape[0]) / X_train.shape[0]
    pred_train = np.zeros(X_train.shape[0])
    pred_test =  np.zeros(X_test.shape[0])
    
    for i in range(iterations):
        # Fit a classifier 
        classifier.fit(X_train, Y_train, w)
        pred_train_i = classifier.predict(X_train)     
        pred_test_i = classifier.predict(X_test)
        
        # output of .predict: 0 or 1
        wrongPrediction = [int(P) for P in (pred_train_i != Y_train)]

        # output of Adaboost \in {-1,1} 
        predVect = [s if s==1 else -1 for s in wrongPrediction]   
        
        # error rate
        err = np.dot(w,wrongPrediction) / w.sum()
        
        # alpha
        alpha = 0.5 * np.log( (1.0 - err) / float(err))
        
        # new weights
        w = np.multiply(w, np.exp([float(i) * alpha for i in predVect]))
       
        pred_train = [x.sum() for x in zip(pred_train, 
                                          [x * alpha for x in pred_train_i])]
        pred_test = [x.sum() for x in zip(pred_test, 
                                         [x * alpha for x in pred_test_i])]
    
    pred_train = np.sign(pred_train)
    pred_test = np.sign(pred_test)
    
    error_train= sum(pred_train != Y_train) / float(Y_train.shape[0])
    error_test = sum(pred_test != Y_test) / float(Y_test.shape[0])
    
    return pred_train , pred_test , error_train , error_test


def dataset_split(X,Y,split_rate):
    train_split=int(split_rate*X.shape[0])

    X_train = X[:train_split]
    X_test = X_train[train_split:]
    Y_train = Y[:train_split]
    Y_test = Y[train_split:]
    
    return X_train, Y_train , X_test , Y_test

#### Main

#load the data (features)


# Split into training and test set
[X_train, Y_train , X_test , Y_test] = dataset_split(X,Y,0.8)              

# Fit a simple decision tree first
base_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)


#errors and predictions before adaboost
base_tree.fit(X_train,Y_train)
pred_train = base_tree.predict(X_train)
pred_test = base_tree.predict(X_test)
    
error_train= sum(pred_train != Y_train) / float(Y_train.shape[0])
error_test = sum(pred_test != Y_test) / float(Y_test.shape[0])

training_pred = []
testing_pred = []
train_err_M = []
test_err_M = []

training_pred.append(pred_train)
testing_pred.append(pred_test)
train_err_M.append(error_train)
test_err_M.append(error_test)
    

M = 100     #Number of iterations for adaboost     
for i in range(1, M, 10):    
    [pred_train , pred_test , error_train , 
     error_test ] = Adaboost(X_train , Y_train , X_test, Y_test, i, base_tree)

    training_pred.append(pred_train)
    testing_pred.append(pred_test)
    train_err_M.append(error_train)
    test_err_M.append(error_test)
    

iRange=np.arange(1, M+10, 10)
trainERR,= plt.plot(iRange,train_err_M,'g')
testERR,= plt.plot(iRange,test_err_M,'r')

plt.xlabel('iterations')
plt.ylabel('Errors')
plt.legend([trainERR,testERR], ["Training error","testing error"])
plt.show()     
