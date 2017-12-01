#### Main
#from AdaBoost import *
import feature
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import optimization



# =============================================================================
# Load the data and split, call loadFeature(...) - Shima
# Perform PCA - Meredith
# Return: a dictionary that has the dimensionality reduced dataset - this could
#           later to used in Adaboost training
# =============================================================================
    
def dataset_split(X,Y,valid_split_rate, seed):
    rows = X.shape[0]
    arr = np.arange(rows)
    np.random.seed(seed)
    np.random.shuffle(arr)
    X_shuffle = X[arr]
    Y_shuffle = Y[arr]
    
    validate_split = int(valid_split_rate*X.shape[0])
    X_train = X_shuffle[:validate_split]
    Y_train = Y_shuffle[:validate_split]
    X_validate = X_shuffle[validate_split:]
    Y_validate = Y_shuffle[validate_split:]
    
    return X_train, Y_train , X_validate, Y_validate 


# =============================================================================
# Visualization and save the result
# =============================================================================
def visualize(testresult, Xdata_t):
    reshaped_arr = np.resize(testresult, 
                             (Xdata_t.shape[0], Xdata_t.shape[1]))
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
    return (im, im2)

    

# =============================================================================
# Data Loading
# =============================================================================
################## Training Data
#X = Image.open("/Users/shimanofallah/Dropbox/Small images-1/train_500.tif")
#Y = Image.open("/Users/shimanofallah/Dropbox/Small images-1/train_500_mask.tif")

X = Image.open("/Users/wuwenjun/Documents/UW/CSE 577/image/train_100.tif")
Y = Image.open("/Users/wuwenjun/Documents/UW/CSE 577/image/train_100_mask.tif")
Xdata = np.array(X)

Ydata = np.array(Y).astype(int)
Ydata[Ydata != 0] = 1
Ydata[Ydata == 0] = -1
resized_labels = np.resize(Ydata, (Ydata.shape[0]*Ydata.shape[1],1))
################## Testing Data
X_t = Image.open("/Users/wuwenjun/Documents/UW/CSE 577/image/test_500.tif")
Y_t = Image.open("/Users/wuwenjun/Documents/UW/CSE 577/image/test_500_mask.tif")



Xdata_t = np.array(X_t)
Ydata_t = np.array(Y_t).astype(int)
Ydata_t[Ydata_t != 0] = 1
Ydata_t[Ydata_t == 0] = -1
Y_test = np.resize(Ydata_t, (Ydata_t.shape[0]*Ydata_t.shape[1],1))


# =============================================================================
# Feature Computation
# =============================================================================
#features = feature.computeFeature(Xdata, [3, 15, 8, 4, 2])

feature.computeStructureFeatures(Xdata, True)
feature.computeStructureFeatures(Xdata_t, False)

feature.computeAllPixelFeatures(Xdata, True)
feature.computeAllHaarlikeFeatures(Xdata, True)
feature.computeAllPixelFeatures(Xdata_t, False)
feature.computeAllHaarlikeFeatures(Xdata_t, False)
#resized_features = np.resize(features, 
#                             (features.shape[0]*features.shape[1], features.shape[2]))


#features_t = feature.computeFeature(Xdata_t, [3, 15, 8, 4, 2])
#X_test = np.resize(features_t, 
#                             (features_t.shape[0]*features_t.shape[1], features_t.shape[2]))
  

pixelF_Tr = np.genfromtxt ('/Users/wuwenjun/GitHub/CSE-577-Melanoma-Nuclei-Segmentation/pixelFeature_Tr.csv', delimiter=",")
haarlikeF_Tr = np.genfromtxt ('/Users/wuwenjun/GitHub/CSE-577-Melanoma-Nuclei-Segmentation/HLFeatures_Tr.csv', delimiter=",")
structF_Tr = np.genfromtext('/Users/wuwenjun/GitHub/CSE-577-Melanoma-Nuclei-Segmentation/structFeatures_Tr.csv', delimiter=",")
allFeatures_Tr = np.concatenate((pixelF_Tr, haarlikeF_Tr, structF_Tr), axis = 1)

pixelF_Ts = np.genfromtxt ('/Users/wuwenjun/GitHub/CSE-577-Melanoma-Nuclei-Segmentation/pixelFeature_Ts.csv', delimiter=",")
haarlikeF_Ts = np.genfromtxt ('/Users/wuwenjun/GitHub/CSE-577-Melanoma-Nuclei-Segmentation/HLFeatures_Ts.csv', delimiter=",")
structF_Ts = np.genfromtext('/Users/wuwenjun/GitHub/CSE-577-Melanoma-Nuclei-Segmentation/structFeatures_Ts.csv', delimiter=",")
X_test = np.concatenate((pixelF_Ts, haarlikeF_Ts, structF_Ts), axis = 1)




# ===========================Splitting & Training==============================
# Split into training and validation set
# set seed for shuffle
# PCA reduction and Trining
# =============================================================================
SEED = 928
[X_train, Y_train , X_validate, Y_validate] = dataset_split(allFeatures_Tr,resized_labels,0.9, SEED)      

# reduce feature dimension
[pca_train, Y_train, pca_validate, Y_validate, pca_test, Y_test] = feature.reduceFeatures(X_train, Y_train , X_validate, Y_validate ,X_test , Y_test)      

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