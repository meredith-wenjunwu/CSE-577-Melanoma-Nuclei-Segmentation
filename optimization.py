#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 23:44:30 2017

@author: wuwenjun
"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# =============================================================================
# Adaboost: parameter selection using validation data
# - number of estimator
# - learning rate
# - maximum depth of decision tree
# =============================================================================
def optimizeAdaboost(X_train, Y_train, X_valid, Y_valid, step):
    # not implemented
    learningrate = [0.1, 0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95, 1]
    est = range(10, 500, step)
    scores = np.zeros((len(learningrate), len(est)))
    for i in xrange(len(learningrate)):
        for j in xrange(len(est)):
            bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                 algorithm="SAMME.R",
                                 n_estimators=est[j],
                                 learning_rate = learningrate[i]
                                 )
            bdt.fit(X_train, Y_train.ravel())
            bdt_train = bdt.predict(X_train)
            bdt_val = bdt.predict(X_valid) 
            f1_train = f1_score(Y_train, bdt_train)
            f1_valid = f1_score(Y_valid, bdt_val)
            scores[i,j] = f1_train - f1_valid
            index = np.unravel_index(scores.argmin(), scores.shape)
    learn = index[0]
    estimator = index[1]
    return(learningrate[learn], est[estimator])
    

