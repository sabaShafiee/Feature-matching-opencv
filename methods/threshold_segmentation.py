import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os


def thresholding(img):
    (T, thresh) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh
