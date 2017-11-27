
#### Main
import AdaBoost
import feature
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# =============================================================================
# Calculate Features, concatenate them, return.
# - windowSizeArray could have length 1, 3, 4 or 5
#      - length 1: pixel feature
#      - length 3: pixel feature and adjacent Haar-like features
#      - length 4: pixel feature, adjacent Haar-like features and nonadjacent Haar like feature 1
#      - length 5: pixel feature, adjacent Haar-like features, nonadjacent Haar like feature 1
#                  and nonadjacent Haar like feature 2
#           
#      - [1:4]: Haar-like feature w1, w2, w3, w4
#           5 <= W1 <= 25
#           1 <= W2,W3,W4 <= 8
# - norm (optional) - normalization factor for nonadjacent Haar-like features (default is 1)
# Return: numpy.array of concatenated features; empty numpy array if wrong size is passed in
# =============================================================================
def computeFeature(input, windowSizeArray, norm = 1):
    if (len(windowSizeArray) == 1):
        return feature.pixelFeature(input, windowSizeArray[0])
    elif (len(windowSizeArray) == 3): 
        pixelF = feature.pixelFeature(input, windowSizeArray[0])
        adjacentHF = feature.adjacentHLFeatures(input, windowSizeArray[1], windowSizeArray[2])
        return np.concatenate((pixelF, adjacentHF), axis = 2)
    elif (len(windowSizeArray) == 4):
        pixelF = feature.pixelFeature(input, windowSizeArray[0])
        adjacentHLF = feature.adjacentHLFeatures(input, windowSizeArray[1], windowSizeArray[2])
        nonadjacentHLF = feature.nonadjacentHLFeatures1(input, windowSizeArray[1], 
                                            windowSizeArray[2], windowSizeArray[3], norm)
        return np.concatenate((pixelF,adjacentHLF, nonadjacentHLF), axis = 2)
    elif (len(windowSizeArray) == 5):
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
    
def dataset_split(X,Y,valid_split_rate,test_split_rate):
    rows = X.shape[0]
    arr = np.arange(rows)
    np.random.shuffle(arr)
    X_shuffle = X[rows]
    Y_shuffle = Y[rows]
    
    test_split = int(test_split_rate*X.shape[0])
    validate_split = int((valid_split_rate*test_split*X.shape[0]))
    X_tr = X_shuffle[:test_split]
    Y_tr = Y_shuffle[:test_split]
    X_train = X_tr[:validate_split]
    Y_train = Y_tr[:validate_split]
    X_validate = X_tr[validate_split:]
    Y_validate = Y_tr[validate_split:]
    X_test = X_shuffle[test_split:]
    Y_test = Y_shuffle[test_split:]
    
    return X_train, Y_train , X_validate, Y_validate ,X_test , Y_test
    
    # Compute Features, I assume:
    # X_train is training data, Y_train is training label
    # X_validate is validation data, Y_validate is validation label
    # X_test is test data, Y_test is test label
    # the window sizes are just examples here. 
#    feature_train = computeFeature(X_train, [3, 15, 8, 4, 2])
#    feature_validate = computeFeature(X_validate, [3, 15, 8, 4, 2])
#    feature_test = computeFeature(X_test, [3, 15, 8, 4, 2])
    # The following are PCA Feature Reduction, threshold are set to be 0.99 for now
def reduceFeatures(feature_train, Y_train, feature_validate, Y_validate, 
                   feature_test, Y_test):
    (pca, numComponents) = doPCA(feature_train, 0.99)
    pca_train = pca.transform(feature_train)
    pca_validate = pca.transform(feature_validate)
    pca_test = pca.transform(feature_test)
    return pca_train, Y_train, pca_validate, Y_validate, pca_test, Y_test



# =============================================================================
# Perform PCA dimensionality reduction:
#   Increase number of components until reaching the threshold of explained
#   variance ratio
# =============================================================================
def doPCA(data, threshold):
    fTotal = data.shape[2]
    for i in xrange(1,fTotal):
        pca = PCA(n_components = i)
        pca.fit(data)
        if (pca.explained_variance_ratio_ > threshold):
            return (pca, i)
    return (pca, fTotal)

X = Image.open("/Users/wuwenjun/Documents/UW/CSE 577/image/test_500.tif")
Y = Image.open("/Users/wuwenjun/Documents/UW/CSE 577/image/test_500_mask.tif")
Xdata = np.array(X)
Ydata = np.array(Y)
# compute the features and change dimensions
features = computeFeature(Xdata, [3, 15, 8, 4, 2])
resized_features = np.resize(features, 
                             (features.shape[0]*features.shape[1], features.shape[2]))
resized_labels = np.resize(Ydata, (Ydata.shape[0]*Ydata.shape[1],1))


# Split into training and test set
[X_train, Y_train , X_validate, Y_validate ,X_test , Y_test] = dataset_split(resized_features,
resized_labels,0.9,0.8)        
# reduce feature dimension
[pca_train, Y_train, pca_validate, Y_validate, pca_test, Y_test] = reduceFeatures(X_train, Y_train , X_validate, Y_validate ,X_test , Y_test)      

# Fit a simple decision tree first
base_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)


#errors and predictions before adaboost
base_tree.fit(pca_train,Y_train)
pred_train = base_tree.predict(pca_train)
pred_val = base_tree.predict(pca_train)
    
error_train= sum(pred_train != pca_train) / float(Y_train.shape[0])
error_val = sum(pred_val != pca_validate) / float(Y_validate.shape[0])

training_pred = []
validation_pred = []
train_err_M = []
val_err_M = []

training_pred.append(pred_train)
validation_pred.append(pred_val)
train_err_M.append(error_train)
val_err_M.append(error_val)
    
##(Shima) should put a stopping point
##(Shima) should change testings with validation and add testing oart at the end
stop = False
epsilon = 0.01
Ada_iter = 0     #Number of iterations for adaboost     
while(stop == False):
    Ada_iter = Ada_iter +1
    past_error_val = error_val.copy()
    [pred_train , pred_val , error_train , error_val ] = Adaboost(pca_train , Y_train , pca_validate, Y_validate, past_error_val, base_tree)
    training_pred.append(pred_train)
    validation_pred.append(pred_val)
    train_err_M.append(error_train)
    val_err_M.append(error_val)
    
    check = np.amax(np.absolute(past_error_val-error_val))
    if check < epsilon:
        stop = True
    

iRange=np.arange(1, Ada_iter+10, 10)
trainERR,= plt.plot(iRange,train_err_M,'g')
valERR,= plt.plot(iRange,val_err_M,'r')

plt.xlabel('iterations')
plt.ylabel('Errors')
plt.legend([trainERR,valERR], ["Training error","Validation error"])
plt.show() 
