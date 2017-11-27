import numpy as np

def Adaboost(X_train , Y_train , X_test, Y_test, past_error, classifier,stopEps):

    # Initialize
    w = np.ones(X_train.shape[0]) / X_train.shape[0]
    
    pred_train = np.zeros(X_train.shape[0])
    pred_test =  np.zeros(X_test.shape[0])
    
    training_pred = []
    testing_pred = []
    train_err_M = []
    test_err_M = []
    error_test = past_error
    stop = False
    Ada_iter = 0
    
    while(stop == False):
        Ada_iter = Ada_iter +1
        past_error_test = error_test.copy()
        
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
    
    
        training_pred.append(pred_train)
        testing_pred.append(pred_test)
        train_err_M.append(error_train)
        test_err_M.append(error_test)
    
        check = np.amax(np.absolute(past_error_test-error_test))
        if check < stopEps:
            stop = True
    
    ### Do we want all the errror on iterations or only the last one?
    return pred_train , pred_test , error_train , error_test , Ada_iter
