import numpy as np
from sklearn.metrics import accuracy_score

def AdaBoost(X_train , Y_train , X_test, Y_test, M, classifier):
    # Initialize
    w = np.ones((X_train.shape[0],1)) / X_train.shape[0]
    
    pred_train = np.zeros((X_train.shape[0],1))
    pred_test =  np.zeros((X_test.shape[0],1))
    
    for i in range(M):
        classifier.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = np.zeros((Y_train.shape[0],1))
        pred_test_i = np.zeros((Y_test.shape[0],1))
        
        pred_train_i[:,0] = classifier.predict(X_train)     
        pred_test_i[:,0] = classifier.predict(X_test)
        
#        wrongPrediction = [int(P) for P in (pred_train_i != Y_train)]
        wrongPrediction = (pred_train_i != Y_train).astype(int)
        
        predVect = [s if s==1 else -1 for s in wrongPrediction]   
        
        err = np.dot(w,wrongPrediction) / sum(w)
        
        alpha = 0.5 * np.log( (1.0 - err) / float(err))
        
        w = np.multiply(w, np.exp([float(i) * alpha for i in predVect]))
       
        pred_train = [sum(x) for x in zip(pred_train, 
                                          [x * alpha for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test, 
                                         [x * alpha for x in pred_test_i])]
    
    
    pred_train = np.sign(pred_train)
    pred_test = np.sign(pred_test)
    error_train= 1 - accuracy_score(Y_train, pred_train)
    error_test = 1 - accuracy_score(Y_test, pred_test)
    
    return error_train, error_test

