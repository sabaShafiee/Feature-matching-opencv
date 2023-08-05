import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import time


dataset_path = r"data"


def image_detect_and_compute(detector, img_name):
    """Detect and compute interest points and their descriptors."""
    img = cv2.imread(os.path.join(dataset_path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(img, None)
    return img, kp, des


def draw_image_matches(detector, img1_name, img2_name, nmatches=10):
    start_time = time.time()
    """Draw ORB feature matches of the given two images."""
    img1, kp1, des1 = image_detect_and_compute(detector, img1_name)
    img2, kp2, des2 = image_detect_and_compute(detector, img2_name)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    MIN_MATCH_COUNT = 10
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        d = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Jumlah Key Point : {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2,
    )
    res = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    end_time = time.time()
    process_time = end_time - start_time
    print("processing time: {:.15f} seconds".format(process_time))

    plt.figure(figsize=(16, 16))
    plt.title(type(detector))
    plt.imshow(res)
    plt.show()


akaze = cv2.AKAZE_create()
draw_image_matches(akaze, "waffle2.jpg", "waffle.jpg")
draw_image_matches(akaze, "building_2.jpg", "building_1.jpg")
