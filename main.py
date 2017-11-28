#### Main
#from AdaBoost import *
import feature
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
#import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# =============================================================================
# Load the data and split, call loadFeature(...) - Shima
# Perform PCA - Meredith
# Return: a dictionary that has the dimensionality reduced dataset - this could
#           later to used in Adaboost training
# =============================================================================
    
def dataset_split(X,Y,valid_split_rate):
    rows = X.shape[0]
    arr = np.arange(rows)
    SEED = 928
    np.random.seed(SEED)
    np.random.shuffle(arr)
    X_shuffle = X[arr]
    Y_shuffle = Y[arr]
    
    validate_split = int(valid_split_rate*X.shape[0])
    X_train = X_shuffle[:validate_split]
    Y_train = Y_shuffle[:validate_split]
    X_validate = X_shuffle[validate_split:]
    Y_validate = Y_shuffle[validate_split:]
    
    return X_train, Y_train , X_validate, Y_validate 
    
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
    fTotal = data.shape[1]
    for i in xrange(1,fTotal):
        pca = PCA(n_components = i)
        pca.fit(data)
        if (sum(pca.explained_variance_ratio_) > threshold):
            return (pca, i)
    return (pca, fTotal)

#X = Image.open("/Users/shimanofallah/Dropbox/Small images-1/train_500.tif")
#Y = Image.open("/Users/shimanofallah/Dropbox/Small images-1/train_500_mask.tif")

X = Image.open("/Users/wuwenjun/Documents/UW/CSE 577/image/train_500.tif")
Y = Image.open("/Users/wuwenjun/Documents/UW/CSE 577/image/train_500_mask.tif")

Xdata = np.array(X)
Ydata = np.array(Y)

# compute the features and change dimensions
features = computeFeature(Xdata, [3, 15, 8, 4, 2])
resized_features = np.resize(features, 
                             (features.shape[0]*features.shape[1], features.shape[2]))
resized_labels = np.resize(Ydata, (Ydata.shape[0]*Ydata.shape[1],1))

# Split into training and validation set
[X_train, Y_train , X_validate, Y_validate] = dataset_split(resized_features,resized_labels,0.9)      

################## Testing Data
X_t = Image.open("/Users/wuwenjun/Documents/UW/CSE 577/image/test_500.tif")
Y_t = Image.open("/Users/wuwenjun/Documents/UW/CSE 577/image/test_500_mask.tif")
Xdata_t = np.array(X_t)
Ydata_t = np.array(Y_t)
features_t = computeFeature(Xdata_t, [3, 15, 8, 4, 2])
X_test = np.resize(features_t, 
                             (features_t.shape[0]*features_t.shape[1], features_t.shape[2]))
Y_test = np.resize(Ydata_t, (Ydata_t.shape[0]*Ydata_t.shape[1],1))
  
# reduce feature dimension
[pca_train, Y_train, pca_validate, Y_validate, pca_test, Y_test] = reduceFeatures(X_train, Y_train , X_validate, Y_validate ,X_test , Y_test)      

# Try built-in decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME.R",
                         n_estimators=30)

bdt.fit(pca_train, Y_train.ravel())
bdt_train = bdt.predict(pca_train)
bdt_val = bdt.predict(pca_validate) 
bdt_test = bdt.predict(pca_test)
error_train = 1 - accuracy_score(Y_train, bdt_train)
error_val = 1 - accuracy_score(Y_validate, bdt_val)
error_test = 1 - accuracy_score(Y_test, bdt_test) 

# =============================================================================
# Visualization
# =============================================================================
# Reshape to image size
reshaped_arr = np.resize(bdt_test, 
                             (Xdata.shape[0], Xdata.shape[1]))
im = Image.fromarray(reshaped_arr)
im.save("/Users/wuwenjun/GitHub/CSE-577-Melanoma-Nuclei-Segmentation/output.jpeg")
#Convert reshaped array to binary mask
reshaped_arr = reshaped_arr != 0
layer1 = Xdata_t[:,:,0]
layer2 = Xdata_t[:,:,1]
layer3 = Xdata_t[:,:,2]
layer1[reshaped_arr] = 0
layer2[reshaped_arr] = 0
layer3[reshaped_arr] = 0
overlay = np.zeros((Xdata.shape[0], Xdata.shape[1], Xdata.shape[2]),'uint8')
overlay[:,:,0] = layer1
overlay[:,:,1] = layer2
overlay[:,:,2] = layer3
im2 = Image.fromarray(overlay, 'RGB')
im2.save("/Users/wuwenjun/GitHub/CSE-577-Melanoma-Nuclei-Segmentation/output_overlay.jpeg")

#
## Fit a simple decision tree first
#base_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)
#
#base_tree.fit(pca_train,Y_train)
#pred_train = base_tree.predict(pca_train)
#pred_val = base_tree.predict(pca_validate)   
#error_train = 1 - accuracy_score(Y_train, pred_train)
#error_val = 1 - accuracy_score(Y_validate, pred_val)
#
#
#
#er_train = [error_train]
#er_val = [error_val]
#iteration=50
#
#for i in range(10, iteration+10, 10):
#    [error_train, error_val] = AdaBoost(pca_train ,Y_train , pca_validate, Y_validate, i, base_tree)
#    er_train.append(error_train)
#    er_val.append(error_val)
#
#
#D = range(0, iteration+10, 10)    
#TR,= plt.plot(D,er_train)
#TE,= plt.plot(D,er_val)
#plt.legend([TR,TE], ["Training data","Testing data"])
#plt.xlabel('iteration')
#plt.ylabel('Error')
#plt.show()