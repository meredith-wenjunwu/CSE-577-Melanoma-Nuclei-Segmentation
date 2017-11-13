from PIL import Image
import numpy as np
import sys
from itertools import islice

def pixelFeature(input_image, windows):
    # windows should be an array of window sizes: 3, 5, 9, 15, 19, 25
    if not input_image:
        print("Input Image is empty!")
        return
    features = np.zeros((input_image.shape[0], input_image.shape[1],
                         4 * len(windows)))
    
    # starting from the top
    for i in xrange(0, len(windows)):
        w = windows[i]
        (winW, winH) = (w, w)
        for (x, y, window) in sliding_window(input_image, (w, w)):
            # if the window size does not meet our desired window size, ignore
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            # perform feature extraction
            # Sum of a window of w * w pixels
            features[y, x, i * 4] = sum(window)
            # Dynamic range of a window of w * w pixels
            features[y, x, i * 4 + 1] = abs(np.amax(window) - np.amin(window))
            # Variance of a window of w * w pixels
            features[y, x, i * 4 + 2] = np.var(window)
            # Median of a window of w * w pixels
            features[y, x, i * 4 + 3] = np.median(window)
    return features

def sliding_window(image, windowSize):
    # slide a window across the image
    if not image:
        print("Input Image is empty!")
        return
    for y in xrange(0, image.shape[0], 1):
        for x in xrange(0, image.shape[1], 1):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def adjacentHLFeatures(input_image, window1, window2):
    # 5 <= W1 <= 25
    # 1 <= W2 <= 8
    if not input_image:
        print("Input Image is empty!")
        return
    features = np.zeros(input_image.shape[0], input_image.shape[1],
                        len(window1) * len(window2))
    for i in xrange(0, len(window1)):
        for j in xrange(0, len(window2)):
            w1 = window1[i]
            w12 = w1 + 2 * window2[i]
            for (x2, y2, window2) in sliding_window(input_image, (w12, w12)):
                window1 = input_image[y2:y2 + w1, x2:x2 + w1]
                features[y2, x2, i * len(window1) + j] = sum(window2)
                - sum(window1)
    return features

def nonadjacentHLFeatures1(input_image, window1, window2, window3, norm):
    # 5 <= W1 <= 25
    # 1 <= W2,W3,W4 <= 8
    # norm is normalization factor, default = 1
    
    if not input_image:
        print("Input Image is empty!")
        return
    len1 = len(window1)
    len2 = len(window2)
    len3 = len(window3)

    features = np.zeros(input_image.shape[0], input_image.shape[1],
                        len1 * len2 * len3)

    for i in xrange(0, len1):
        for j in xrange(0, len2):
            for k in xrange(0, len3):
                w1 = window1[i]
                w2 = window2[j]
                w3 = window3[k]
                w12 = w1 + 2 * w2
                w123 = w1 + 2 * (w2 + w3)
                for (x123, y123, window123) in sliding_window(input_image, (w123, w123)):
                    window12 = input_image[y123:y123 + w12, x123:x123 + w12]
                    window1 = input_image[y123:y123 + w1, x123:x123 + w1]
                    # HLNA(w1, w2, w3)
                    features[y123, x123, i * len1 + j * len2 + k] = sum(window123)
                    - sum(window12) - norm * sum(window1)
    return features

def nonadjacentHLFeatures2(input_image, window1, window2, window3, window4, norm):
    # 5 <= W1 <= 25
    # 1 <= W2,W3,W4 <= 8
    # norm is normalization factor, default = 1
    
    if not input_image:
        print("Input Image is empty!")
        return
    len1 = len(window1)
    len2 = len(window2)
    len3 = len(window3)
    len4 = len(window4)

    features = np.zeros(input_image.shape[0], input_image.shape[1],
                        len1 * len2 * len3 * len4)

    for i in xrange(0, len1):
        for j in xrange(0, len2):
            for k in xrange(0, len3):
                for l in xrange(0, len4):
                    w1 = window1[i]
                    w2 = window2[j]
                    w3 = window3[k]
                    w4 = window4[l]
                    w12 = w1 + 2 * w2
                    w123 = w1 + 2 * (w2 + w3)
                    w1234 = w123 + 2 * w4
                    for (x1234, y1234, window1234) in sliding_window(input_image, (w1234, w1234)):
                        window123 = input_image[y1234:y1234 + w123, x1234:x1234 + w123]
                        window12 = input_image[y1234:y1234 + w12, x1234:x1234 + w12]
                        window1 = input_image[y1234:y1234 + w1, x1234:x1234 + w1]
                        # HLNA2(w1, w2, w3, w4)
                        index = i * len1 + j * len2 + k * len3 + l
                        features[y1234, x1234, index] = sum(window1234)
                        - sum(window123) - norm * sum(window12) - sum(window1)
    return features
