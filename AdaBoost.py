#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 6 18:26:58 2017

@author: shima
"""

import numpy as np

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

