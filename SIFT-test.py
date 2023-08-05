# impoert libraries

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import time
from methods.SIFT import draw_image_matches
from super_resolution import LapSRN_sResolution, ESPCN_sResolution


def open_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def increas_resolution(map_img):
    # map increase resolution
    res_img_map, processtime = ESPCN_sResolution(img_map, saved=False)
    res_path = r"res_output\res_img_map.jpg"
    cv2.imwrite(res_path, res_img_map)
    print("[INFO] super resolution took {:.6f} seconds".format(processtime))
    print(
        "[INFO] map image size is {} * {}".format(
            res_img_map.shape[0], res_img_map.shape[1]
        )
    )

    return res_img_map


segment_methods = [
    "kmeans",
    "canny",
    "threshold",
    "laplacian",
    "gradient",
    "meanshift",
    "region",
]

# image opening
image_map_path = r"E:\dataset\satellite-data\frame-1000-2.jpg"
image_cam_path = r"E:\dataset\data\frame-1000.jpg"

img_cam = open_image(image_cam_path)
img_map = open_image(image_map_path)


# images preprocessing for extracting better features

# see the size
print("[INFO] camera image size is {} * {}".format(img_cam.shape[0], img_cam.shape[1]))
print("[INFO] map image size is {} * {}".format(img_map.shape[0], img_map.shape[1]))

# cropped_cam_img = img_cam[300:1800, 500:3000]
# cropped_map_img = img_map[50:800, 150:1600]
img_map = increas_resolution(img_map)

# image feature matching
matcher = cv2.SIFT_create()
draw_image_matches(matcher, img_cam, img_map)
