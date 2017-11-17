#### Main
import Adaboost
import feature
import numpy as np
from sklearn.decoposition import PCA


# =============================================================================
#  Calculate Features, concatenate them, return.
#  windowSizeArray should have length of 5
#      - [0]: pixel feature
#      - [1:4]: Haar-like feature w1, w2, w3, w4
#      # 5 <= W1 <= 25
#      1 <= W2,W3,W4 <= 8
#  norm (optional) - normalization factor for nonadjacent Haar-like features (default is 1)
# =============================================================================
def computeFeature(input, windowSizeArray, norm = 1):
    if (len(windowSizeArray == 1)):
        return feature.pixelFeature(input, windowSizeArray[0])
    elif (len(windowSizeArray == 3)): 
        pixelF = feature.pixelFeature(input, windowSizeArray[0])
        adjacentHF = feature.adjacentHLFeatures(input, windowSizeArray[1], windowSizeArray[2])
        return np.concatenate((pixelF, adjacentHF), axis = 2)
    elif (len(windowSizeArray == 4)):
        pixelF = feature.pixelFeature(input, windowSizeArray[0])
        adjacentHLF = feature.adjacentHLFeatures(input, windowSizeArray[1], windowSizeArray[2])
        nonadjacentHLF = feature.nonadjacentHLFeatures1(input, windowSizeArray[1], 
                                            windowSizeArray[2], windowSizeArray[3], norm)
        return np.concatenate((pixelF,adjacentHLF, nonadjacentHLF), axis = 2)
    elif (len(windowSizeArray == 5)):
        pixelF = feature.pixelFeature(input, windowSizeArray[0])
        adjacentHLF = feature.adjacentHLFeatures(input, windowSizeArray[1], windowSizeArray[2])
        nonadjacentHLF = feature.nonadjacentHLFeatures1(input, windowSizeArray[1], 
                                            windowSizeArray[2], windowSizeArray[3], norm)
        nonadjacentHLF2 = feature.nonadjacentHLFeatures2(input, windowSizeArray[1], windowSizeArray[2], 
                                             windowSizeArray[3], windowSizeArray[4], norm)
        return np.concatenate((pixelF,adjacentHLF, nonadjacentHLF, nonadjacentHLF2), axis = 2)
    else: 
        # Invalid Size, return empty array
        print("Not enough or too much window size provided. Returning empty np array")
        return np.array([]) 
   
    
# =============================================================================
# Load the data and split, call loadFeature(...) - Shima
# Perform PCA - Meredith
# Return: a dictionary that has the dimensionality reduced dataset - this could
#           later to used in Adaboost training
# =============================================================================
def loadData(addresses):
    # data loading not implemented
    
    
    # Compute Features, I assume:
    # X_train is training data, Y_train is training label
    # X_validate is validation data, Y_validate is validation label
    # X_test is test data, Y_test is test label
    # the window sizes are just examples here. 
    feature_train = computeFeature(X_train, [3, 15, 8, 4, 2])
    feature_validate = computeFeature(X_validate, [3, 15, 8, 4, 2])
    feature_test = computeFeature(X_test, [3, 15, 8, 4, 2])
    # The following are PCA Feature Reduction, threshold are set to be 0.99 now
    (pca, numComponents) = doPCA(feature_train, 0.99)
    pca_train = pca.transform(feature_train)
    pca_validate = pca.transform(feature_validate)
    pca_test = pca.transform(feature_test)
    return {'data_train': pca_train, 'label_train': Y_train, 
            'data_validate': pca_validate, 'label_validate': Y_validate,
            'data_test': pca_test, 'label_test': Y_test}



# =============================================================================
# Increase number of components until reaching the threshold of explained
# variance ratio
# =============================================================================

def doPCA(data, threshold):
    fTotal = data.shape[2]
    for i in xrange(1,fTotal):
        pca = PCA(n_components = i)
        pca.fit(data)
        if (pca.explained_variance_ratio_ > treshold):
            return (pca, i)


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
