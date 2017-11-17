from PIL import Image
import numpy as np
import sys
from itertools import islice
import random

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
def nonadjacentHLFeatures1(input_image, window1, window2, window3, norm):
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

def nonadjacentHLFeatures2(input_image, window1, window2, window3, window4, norm):
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
