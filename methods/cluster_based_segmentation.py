import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.cluster import KMeans
import time


def kmean_segment(img):
    # convert img to gray
    start = time.time()
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    # print(pixel_values.shape)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3
    _, labels, (centers) = cv2.kmeans(
        pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    segmented_image = segmented_image.reshape(img.shape)

    end = time.time()
    processing_time = end - start
    print("processing time is {} seconds.".format(processing_time))
    print("number of clusters is {} .".format(k))
    return segmented_image
