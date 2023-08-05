import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import time
from methods.cluster_based_segmentation import kmean_segment
from methods.edge_segmentation import canny_seg, gradient
from methods.threshold_segmentation import thresholding


segment_methods = [
    "kmeans",
    "canny",
    "threshold",
    "gradient",
    "meanshift",
    "region",
]


def segmenting_image(img, method):
    if method == "kmeans":
        img = kmean_segment(img)

    if method == "canny":
        img = canny_seg(img)

    if method == "threshold":
        img = thresholding(img)

    if method == "gradient":
        img = gradient(img)

    return img


def image_detect_and_compute(detector, image):
    """Detect and compute interest points and their descriptors."""
    image = segmenting_image(image, segment_methods[3])
    kp, des = detector.detectAndCompute(image, None)
    return kp, des


def draw_image_matches(detector, img1, img2, nmatches=10):
    start_time = time.time()
    kp1, des1 = image_detect_and_compute(detector, img1)
    kp2, des2 = image_detect_and_compute(detector, img2)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    good_without_list = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
            good_without_list.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_without_list]).reshape(
        -1, 1, 2
    )
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_without_list]).reshape(
        -1, 1, 2
    )
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    matchesMask = mask.ravel().tolist()
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2,
    )

    res = cv2.drawMatchesKnn(
        img1,
        kp1,
        img2,
        kp2,
        good,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    end_time = time.time()
    process_time = end_time - start_time
    print("processing time: {:.15f} seconds".format(process_time))

    plt.figure(figsize=(16, 16))
    plt.title(type(detector))
    plt.imshow(res)
    plt.show()
