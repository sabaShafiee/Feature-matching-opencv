import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os


def laplasian(img):
    blured_img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Laplacian(blured_img, cv2.CV_16S, ksize=3)
    edges = cv2.convertScaleAbs(edges)
    return edges
    # Display Sobel Edge Detection Images


def canny_seg(img):
    blured_img = cv2.GaussianBlur(img, (7, 7), 0)
    edges = cv2.Canny(image=blured_img, threshold1=50, threshold2=150)
    return edges


def gradient(img):
    gX = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0)
    gY = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1)
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)

    # combine the gradient representations into a single image

    magnitude = np.sqrt((gX**2) + (gY**2))
    orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
    combined = cv2.addWeighted(magnitude, 0.5, orientation, 0.5, 0)

    return combined
