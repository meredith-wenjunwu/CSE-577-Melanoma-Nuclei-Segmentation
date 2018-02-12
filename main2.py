#### Main
#from AdaBoost import *
import feature
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support
import optimization
import os
import pickle


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
# Splitted Image Feature Computation (Loop)
# folder path should be the folder that contains the splitted iages
# =============================================================================
def computeFeatureInDirectory(inputDirectory = "croppedImages/", outputDirectory = 'FR/'):
    
    if not os.path.exists(outputDirectory):
            os.makedirs(outputDirectory)
    for filename in os.listdir(inputDirectory):
        if filename.endswith(".tif") or filename.endswith(".png"): 
            name, file_extension = os.path.splitext(filename)
            path = os.path.join(inputDirectory, filename)
            print(path)
            X = Image.open(path)
            image = np.array(X)
            outpath = os.path.join(outputDirectory, name)
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            
            if "train" in filename:
                feature.computeStructureFeatures(image, True, outpath)
                feature.computeAllPixelFeatures(image, True, outpath)
                # feature.computeAllHaarlikeFeatures(image, True, outpath)
            else:
                feature.computeStructureFeatures(image, False, outpath)
                feature.computeAllPixelFeatures(image, False, outpath)
                # feature.computeAllHaarlikeFeatures(image, False, outpath)



# =============================================================================
# Data Loading
# =============================================================================
################### Training Data
# X = Image.open("image_merged.tif")
# Y = Image.open("mask_merged.tif")
#Xdata = np.array(X)
#
# Ydata = np.array(Y).astype(int)
# Ydata[Ydata != 0] = 1
# Ydata[Ydata == 0] = -1
# resized_labels = np.resize(Ydata, (Ydata.shape[0]*Ydata.shape[1],1))
################### Testing Data
#X_t = Image.open("test.tif")
#Y_t = Image.open("test_mask.tif")
#
#
#
#Xdata_t = np.array(X_t)
#Ydata_t = np.array(Y_t).astype(int)
#Ydata_t[Ydata_t != 0] = 1
#Ydata_t[Ydata_t == 0] = -1
#Y_test = np.resize(Ydata_t, (Ydata_t.shape[0]*Ydata_t.shape[1],1))


# =============================================================================
# General Feature Computation
# =============================================================================
#feature.computeStructureFeatures(Xdata, True)
#feature.computeStructureFeatures(Xdata_t, False)
#
#feature.computeAllPixelFeatures(Xdata, True)
#feature.computeAllHaarlikeFeatures(Xdata, True)
#feature.computeAllPixelFeatures(Xdata_t, False)
#feature.computeAllHaarlikeFeatures(Xdata_t, False)
computeFeatureInDirectory(inputDirectory='croppedImages/subset')

# ============================ load features===================================
# load features
# =============================================================================
# pixelF_Tr = np.load('FR/pixelFeature_Tr.npz')['data']
#haarlikeF_Tr = np.load('FR/HLFeatures_Tr.npz')['data']
# structF_Tr = np.load('FR/structFeatures_Tr.npz')['data']
# allFeatures_Tr = np.concatenate((pixelF_Tr, structF_Tr), axis = 1)
# allFeatures_Tr = np.concatenate((pixelF_Tr, haarlikeF_Tr, structF_Tr), axis = 1)
# X_train = allFeatures_Tr
# Y_train = resized_labels
#
#pixelF_Ts = np.load('FR/pixelFeature_Ts.npz')['data']
#haarlikeF_Ts = np.load('FR/HLFeatures_Ts.npz')['data']
#structF_Ts = np.load('FR/structFeatures_Ts.npz')['data']
#X_test = np.concatenate((pixelF_Ts, haarlikeF_Ts, structF_Ts), axis = 1)


# ===========================Splitting & Training==============================
# Split into training and validation set
# set seed for shuffle
# PCA reduction and Trining
# =============================================================================
#SEED = 928
#[X_train, Y_train , X_validate, Y_validate] = dataset_split(allFeatures_Tr,resized_labels,0.9, SEED)      
#
## reduce feature dimension
#[pca_train, Y_train, pca_validate, Y_validate, pca_test, Y_test] = feature.reduceFeatures(X_train, Y_train , X_validate, Y_validate ,X_test , Y_test)
# print 'PCA'
# pca_train = feature.reduceFeatureSimple(X_train)

# print 'Build Classifier'
# Try built-in decision tree
# bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
#                         algorithm="SAMME.R",
#                         n_estimators=50)

# bdt.fit(pca_train, Y_train.ravel())
# bdt_train = bdt.predict(pca_train)

# filename = 'classifier/modelwithPixelandStruct.sav'
# pickle.dump(bdt, open(filename, 'wb'))
#bdt_val = bdt.predict(pca_validate) 
#bdt_test = bdt.predict(pca_test)
# error_train = 1 - accuracy_score(Y_train, bdt_train)
# print 'error train: %.4f' %error_train
#error_val = 1 - accuracy_score(Y_validate, bdt_val)
#error_test = 1 - accuracy_score(Y_test, bdt_test) 
#
#### Calculate the metrics
# test_metrics = precision_recall_fscore_support(Y_train, bdt_train)
# print 'Precision = ' , "%.4f" %test_metrics[0][0]
# print 'Recall = ' , "%.4f" %test_metrics[1][0]
# print 'F1 score = ' , "%.4f" %test_metrics[2][0]


# save the model to disk