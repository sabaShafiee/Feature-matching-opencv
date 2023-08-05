import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import time


def image_detect_and_compute(detector, image):
    """Detect and compute interest points and their descriptors."""
    kp, des = detector.detectAndCompute(image, None)
    return kp, des


def draw_image_matches(detector, image1, image2, nmatches=10):
    start_time = time.time()
    """Draw ORB feature matches of the given two images."""
    kp1, des1 = image_detect_and_compute(detector, image1)
    kp2, des2 = image_detect_and_compute(detector, image2)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

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
    h, w = image1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(image2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2,
    )

    res = cv2.drawMatchesKnn(
        image1,
        kp1,
        image2,
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
