#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 02:42:22 2017

@author: wuwenjun
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import measure
from os import path
from os import makedirs

def cellBodyfind(original, binary, directory = 'Output/', 
                 area_threshold_low = 700):
    if not path.exists(directory):
        makedirs(directory)
    if (len(binary.shape) != 2):
        print "Size of image is not Correct!"
        print "Expected dimension is 2, actual dimension is %d" %len(binary.shape)
        return
    print "The shape of binary output: (%d, %d)" %binary.shape
    area_threshold_high = binary.shape[0] * binary.shape[1] / 8
    savepath = path.join(directory,'connected.png')
    labeled, nr_objects = findConnected(binary, savepath)
    newlabeled = labeled
    props = measure.regionprops(labeled)

    for prop in props:
        if prop.area > area_threshold_low and prop.area < area_threshold_high:
            newlabeled, nr_objects = separateConnected(original,labeled, 
                                                       prop, nr_objects, 
                                                       area_threshold_low)
    
    plt.imshow(original)
    newprops = measure.regionprops(newlabeled)
    for prop in newprops:
        if prop.area < area_threshold_high:
            center = prop.centroid
            plt.scatter(center[1], center[0], marker="+", color='r', s = 12)
    plt.axis('off')
    savepath = path.join(directory,'centroidLabeled.png')
    plt.savefig(savepath, bbox_inches='tight', dpi=80 * 10)
    plt.show()
    
    

def findConnected(binary, savepath = 'Output/out.png'):
    print "Finding Connected Components in the image:"
    blur_radius = 3
    threshold = 100
    
    imgf = ndimage.gaussian_filter(binary, blur_radius)
    scipy.misc.imsave('Output/gaussian.png', imgf)
    
    labeled, nr_objects = ndimage.label(imgf > threshold)
    print "Number of connected objects is %d " %nr_objects
    
    
    scipy.misc.imsave(savepath, labeled)
    print "Labeled image saved to  %s" %path
    return labeled, nr_objects
#    plt.imshow(labeled)
#    
#    plt.show()


def separateConnected(original,labeled, prop, nr_objects, area_threshold_low):
    # extract the connected nucleus
    grayscale = np.average(original, axis = 2)
    flag = False
    label = prop.label
    boxInterval = prop.bbox
    boundingbox = labeled[boxInterval[0]:boxInterval[2],
                          boxInterval[1]:boxInterval[3]]
    boundingbox_g = grayscale[boxInterval[0]:boxInterval[2],
                          boxInterval[1]:boxInterval[3]]
#    plt.imshow(boundingbox)
#    plt.show()
    binary = boundingbox == label
    boundingbox_g[~binary] = 0
    distance = ndimage.distance_transform_edt(boundingbox_g)
    local_maxi = peak_local_max(distance, indices=False,
                            labels=binary, 
                            num_peaks = prop.area/area_threshold_low, 
                            min_distance = 3 * prop.area/area_threshold_low + 2)
    
    markers = ndimage.label(local_maxi)[0]
    if len(markers) == 1:
        return labeled,nr_objects
    waterlabeled = watershed(-distance, markers, mask = binary, 
                             compactness = 0.3)
    newprops = measure.regionprops(waterlabeled)
    # if nothing is found, return the original
    if len(newprops) == 1:
        return labeled,nr_objects
    # if there separate components was found:
    for newprop in newprops:
        if (flag):
            nr_objects = nr_objects + 1
            label = nr_objects
        mask = waterlabeled == newprop.label
        boundingbox[mask] = label
        flag = True
    newlabeled = labeled
    newlabeled[boxInterval[0]:boxInterval[2],
                          boxInterval[1]:boxInterval[3]] = boundingbox
    return newlabeled, nr_objects
#    plt.imshow(labels)
#    plt.show()
    
# =============================================================================
# Test Code
# Use a mask from training data    
# =============================================================================

fname1 = 'Result/Mask_IntensityNoMorph.tif'
fname2 = 'Result/test_500.tif'   
mask = scipy.misc.imread(fname1)
original = scipy.misc.imread(fname2)
print(original.shape)
cellBodyfind(original, mask)