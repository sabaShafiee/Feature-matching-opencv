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


def draw_image_matches(detector, img1_name, img2_name, nmatches=15):
    start_time = time.time()
    """Draw ORB feature matches of the given two images."""
    img1, kp1, des1 = image_detect_and_compute(detector, img1_name)
    img2, kp2, des2 = image_detect_and_compute(detector, img2_name)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(
        matches, key=lambda x: x.distance
    )  # Sort matches by distance.  Best come first.

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = img1.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    ## draw found regions
    img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255), 1, cv2.LINE_AA)
    res = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)

    end_time = time.time()
    process_time = end_time - start_time
    print("processing time: {} seconds".format(process_time))

    plt.figure(figsize=(16, 16))
    plt.title(type(detector))
    plt.imshow(res)
    plt.show()


orb = cv2.ORB_create(1000, 2)
draw_image_matches(orb, "waffle2.jpg", "waffle.jpg")
