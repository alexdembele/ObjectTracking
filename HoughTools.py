'''
Created on May 19, 2013

@author: vinnie
@github : https://github.com/vmonaco/general-hough/
'''

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

# Good for the b/w test images used
MIN_CANNY_THRESHOLD = 10
MAX_CANNY_THRESHOLD = 50

def gradient_orientation(image):
    '''
    Calculate the gradient orientation for edge point in the image
    '''
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3 )
    gradient = np.arctan2(dy,dx) * 180 / np.pi

    return gradient

def build_r_table(image, origin):
    '''
    Build the R-table from the given shape image and a reference point
    '''
    edges = cv2.Canny(image, MIN_CANNY_THRESHOLD,
                  MAX_CANNY_THRESHOLD)
    gradient = gradient_orientation(edges)

    r_table = defaultdict(list)
    for (i,j),value in np.ndenumerate(edges):
        if value:
            r_table[gradient[i,j]].append((origin[0]-i, origin[1]-j))

    return r_table

def accumulate_gradients(r_table, grayImage):
    '''
    Perform a General Hough Transform with the given image and R-table
    '''
    edges = cv2.Canny(grayImage, MIN_CANNY_THRESHOLD,
                  MAX_CANNY_THRESHOLD)
    gradient = gradient_orientation(edges)

    accumulator = np.zeros(grayImage.shape)
    non_zero_indices = np.nonzero(edges)

    # Compute accumulation
    for i, j in zip(non_zero_indices[0], non_zero_indices[1]):
        for r in r_table[gradient[i, j]]:
            accum_i, accum_j = i + r[0], j + r[1]
            if 0 <= accum_i < accumulator.shape[0] and 0 <= accum_j < accumulator.shape[1]:
                accumulator[accum_i, accum_j] += 1

    return accumulator

def accumulate_gradients_Window(r_table, grayImage,x_prec,y_prec,w,h):
    '''
    Perform a General Hough Transform with the given image and R-table in a specific window around the position of the object in the last frame
    '''
    edges = cv2.Canny(grayImage, MIN_CANNY_THRESHOLD,
                  MAX_CANNY_THRESHOLD)
    gradient = gradient_orientation(edges)

    accumulator = np.zeros(grayImage.shape)
    x1 = max(0,x_prec - w//2)
    x2 = min(len(grayImage[0]),y_prec + w//2)
    y1 = max(0,x_prec - w//2)
    y2 = min(len(grayImage[1]),y_prec + w//2)

    edges_copy = np.zeros_like(edges)
    edges_copy[x1:x2+1, y1:y2+1] = edges[x1:x2+1, y1:y2+1]

    non_zero_indices = np.nonzero(edges_copy)

    # Compute accumulation
    for i, j in zip(non_zero_indices[0], non_zero_indices[1]):
        for r in r_table[gradient[i, j]]:
            accum_i, accum_j = i + r[0], j + r[1]
            if 0 <= accum_i < accumulator.shape[0] and 0 <= accum_j < accumulator.shape[1]:
                accumulator[accum_i, accum_j] += 1

    return accumulator



def n_max(a, n):
    '''
    Return the N max elements and indices in a
    '''
    indices = a.ravel().argsort()[-n:]
    indices = (np.unravel_index(i, a.shape) for i in indices)
    return [(a[i], i) for i in indices]




