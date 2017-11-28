#### Functions

#from PIL import Image
import numpy as np
#import sys
#from itertools import islice
#import random
from sklearn.decomposition import PCA


# Caluculated pixel-related features with sliding window
# ---sum of pixel values
# ---range of pixel values
# ---variance of pixel values
# ---median of pixel values
def pixelFeature(input_image, w):
    # windows should be an array of window sizes: 3, 5, 9, 15, 19, 25
    if not input_image.size:
        print("Input Image is empty!")
        return
    features = np.zeros((input_image.shape[0], input_image.shape[1], 4 * input_image.shape[2]))
    
    # starting from the top
    (winW, winH) = (w, w)
    layer = input_image.shape[2]
    for (x, y, window) in sliding_window(input_image, (w, w)):
        # if the window size does not meet our desired window size, ignore
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        # perform feature extraction
        # Sum of a window of w * w pixels
        features[y, x, 0:layer] = sum(sum(window))
        # Dynamic range of a window of w * w pixels
        for i in xrange(0, layer):
            features[y, x, i + layer] = abs(np.amax(window[:, :, i]) - np.amin(window[:, :, i]))
            # Variance of a window of w * w pixels
        for i in xrange(0, layer):
            features[y, x, i + 2 * layer] = np.var(window[:, :, i])
        for i in xrange(0, layer):
            # Median of a window of w * w pixels
            features[y, x, i + 3 * layer] = np.median(window[:, :, i])
    return features


def sliding_window(image, windowSize):
    # slide a window across the image
    if not image.size:
        print("Input Image is empty!")
        return
    for y in xrange(0, image.shape[0], 1):
        for x in xrange(0, image.shape[1], 1):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def adjacentHLFeatures(input_image, window1, window2):
    # 5 <= W1 <= 25
    # 1 <= W2 <= 8
    if not input_image.size:
        print("Input Image is empty!")
        return
    features = np.zeros((input_image.shape[0], input_image.shape[1], input_image.shape[2]))
    w1 = window1
    w12 = w1 + 2 * window2
    for (x2, y2, window2) in sliding_window(input_image, (w12, w12)):
        window1 = input_image[y2:y2 + w1, x2:x2 + w1]
        features[y2, x2, 0:input_image.shape[2]] = sum(sum(window2)) - sum(sum(window1))
    return features


# Caluculated nonadjacent Haar-Like Features
def nonadjacentHLFeatures1(input_image, window1, window2, window3, norm = 1):
    # 5 <= W1 <= 25
    # 1 <= W2,W3,W4 <= 8
    # norm is normalization factor, default = 1
    
    if not input_image.size:
        print("Input Image is empty!")
        return
 
    features = np.zeros(input_image.shape)

    w1 = window1
    w2 = window2
    w3 = window3
    w12 = w1 + 2 * w2
    w123 = w1 + 2 * (w2 + w3)
    for (x123, y123, window123) in sliding_window(input_image, (w123, w123)):
        window12 = input_image[y123:y123 + w12, x123:x123 + w12]
        window1 = input_image[y123:y123 + w1, x123:x123 + w1]
        # HLNA(w1, w2, w3)
        features[y123, x123, 0:input_image.shape[2]] = sum(sum(window123))
        - sum(sum(window12)) - norm * sum(sum(window1))
    return features


# Caluculated nonadjacent Haar-Like Features 2
def nonadjacentHLFeatures2(input_image, window1, window2, window3, window4, norm = 1):
    # 5 <= W1 <= 25
    # 1 <= W2,W3,W4 <= 8
    # norm is normalization factor, default = 1
    
    if not input_image.size:
        print("Input Image is empty!")
        return

    features = np.zeros(input_image.shape)
    w1 = window1
    w2 = window2
    w3 = window3
    w4 = window4
    w12 = w1 + 2 * w2
    w123 = w1 + 2 * (w2 + w3)
    w1234 = w123 + 2 * w4
    for (x1234, y1234, window1234) in sliding_window(input_image, (w1234, w1234)):
        window123 = input_image[y1234:y1234 + w123, x1234:x1234 + w123]
        window12 = input_image[y1234:y1234 + w12, x1234:x1234 + w12]
        window1 = input_image[y1234:y1234 + w1, x1234:x1234 + w1]
        # HLNA2(w1, w2, w3, w4)

        features[y1234, x1234, 0:input_image.shape[2]] = sum(sum(window1234))
        - sum(sum(window123)) - norm * sum(sum(window12))
        - sum(sum(window1))
    return features


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
        return pixelFeature(input, windowSizeArray[0])
    elif (len(windowSizeArray) == 3): 
        pixelF = pixelFeature(input, windowSizeArray[0])
        adjacentHF = adjacentHLFeatures(input, windowSizeArray[1], windowSizeArray[2])
        return np.concatenate((pixelF, adjacentHF), axis = 2)
    elif (len(windowSizeArray) == 4):
        adjacentHLF = adjacentHLFeatures(input, windowSizeArray[0], windowSizeArray[1])
        nonadjacentHLF = nonadjacentHLFeatures1(input, windowSizeArray[0], 
                                            windowSizeArray[1], windowSizeArray[2], norm)
        nonadjacentHLF2 = nonadjacentHLFeatures2(input, windowSizeArray[0], windowSizeArray[1], 
                                             windowSizeArray[2], windowSizeArray[3], norm)
        return np.concatenate((pixelF,adjacentHLF, nonadjacentHLF, nonadjacentHLF2), axis = 2)
    elif (len(windowSizeArray) == 5):
        pixelF = pixelFeature(input, windowSizeArray[0])
        adjacentHLF = adjacentHLFeatures(input, windowSizeArray[1], windowSizeArray[2])
        nonadjacentHLF = nonadjacentHLFeatures1(input, windowSizeArray[1], 
                                            windowSizeArray[2], windowSizeArray[3], norm)
        nonadjacentHLF2 = nonadjacentHLFeatures2(input, windowSizeArray[1], windowSizeArray[2], 
                                             windowSizeArray[3], windowSizeArray[4], norm)
        return np.concatenate((pixelF,adjacentHLF, nonadjacentHLF, nonadjacentHLF2), axis = 2)
    else: 
        # Invalid Size, return empty array
        print("Not enough or too much window size provided. Returning empty np array")
        return np.array([]) 


# =============================================================================
# Reduce the number of features using PCA
# =============================================================================
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

# def main():
#     im = Image.open('/Users/wuwenjun/Downloads/sample.jpg')
#     imdata = np.array(im.getdata(), dtype=np.float64).reshape(im.size[1], im.size[0], 3)
#     print imdata.shape
    #pFeature = pixelFeature(imdata, 3)
    #adjacentF = adjacentHLFeatures(imdata, 3, 1)
    #print pFeature.shape
    #print adjacentF.shape
    #nonadjF1 = nonadjacentHLFeatures1(imdata, 5, 2, 1, 1)
    #nonadjF2 = nonadjacentHLFeatures2(imdata, 10, 8, 4, 2, 1)

    # with file('test.txt', 'w') as outfile:  
    #     outfile.write('# Array shape: {0}\n'.format(nonadjF2.shape))
    #     for data_slice in nonadjF2:
    #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
    #         outfile.write('# New slice\n')

            
# main()
