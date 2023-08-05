# importing libraries
import time
import cv2
import os
import argparse


def EDSR_sResolution(image):
    modelPath = r"EDSR_x4.pb"

    modelName = modelPath.split(os.path.sep)[-1].split("_")[0].lower()
    modelScale = modelPath.split("_x")[-1]
    modelScale = int(modelScale[: modelScale.find(".")])

    # initialize OpenCV's super resolution DNN object, load the super
    # resolution model from disk, and set the model name and scale
    print("[INFO] loading super resolution model: {}".format(modelPath))
    print("[INFO] model name: {}".format(modelName))
    print("[INFO] model scale: {}".format(modelScale))

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(modelPath)
    sr.setModel(modelName, modelScale)

    start = time.time()
    upscaled = sr.upsample(image)
    end = time.time()

    outputPath = r"res_output\EDSR_output.jpg"
    cv2.imwrite(outputPath, upscaled)

    return (upscaled, end - start)


def ESPCN_sResolution(image, saved=True):
    modelPath = r"models\ESPCN_x4.pb"

    modelName = modelPath.split(os.path.sep)[-1].split("_")[0].lower()
    modelScale = modelPath.split("_x")[-1]
    modelScale = int(modelScale[: modelScale.find(".")])

    # initialize OpenCV's super resolution DNN object, load the super
    # resolution model from disk, and set the model name and scale
    print("[INFO] loading super resolution model: {}".format(modelPath))
    print("[INFO] model name: {}".format(modelName))
    print("[INFO] model scale: {}".format(modelScale))

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(modelPath)
    sr.setModel(modelName, modelScale)

    start = time.time()
    upscaled = sr.upsample(image)
    end = time.time()

    if saved:
        outputPath = r"res_output\ESPCN_output.jpg"
        cv2.imwrite(outputPath, upscaled)

    return (upscaled, end - start)


def FSRCNN_sResolution(image):
    modelPath = r"models\FSRCNN_x4.pb"

    modelName = modelPath.split(os.path.sep)[-1].split("_")[0].lower()
    modelScale = modelPath.split("_x")[-1]
    modelScale = int(modelScale[: modelScale.find(".")])

    # initialize OpenCV's super resolution DNN object, load the super
    # resolution model from disk, and set the model name and scale
    print("[INFO] loading super resolution model: {}".format(modelPath))
    print("[INFO] model name: {}".format(modelName))
    print("[INFO] model scale: {}".format(modelScale))

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(modelPath)
    sr.setModel(modelName, modelScale)

    start = time.time()
    upscaled = sr.upsample(image)
    end = time.time()

    outputPath = r"res_output\FSRCNN_output.jpg"
    cv2.imwrite(outputPath, upscaled)

    return (upscaled, end - start)


def LapSRN_sResolution(image, saved=True):
    modelPath = r"models\LapSRN_x8.pb"

    modelName = modelPath.split(os.path.sep)[-1].split("_")[0].lower()
    modelScale = modelPath.split("_x")[-1]
    modelScale = int(modelScale[: modelScale.find(".")])

    # initialize OpenCV's super resolution DNN object, load the super
    # resolution model from disk, and set the model name and scale
    print("[INFO] loading super resolution model: {}".format(modelPath))
    print("[INFO] model name: {}".format(modelName))
    print("[INFO] model scale: {}".format(modelScale))

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(modelPath)
    sr.setModel(modelName, modelScale)

    start = time.time()
    upscaled = sr.upsample(image)
    end = time.time()

    if saved:
        outputPath = r"res_output\LapSRN_output.jpg"
        cv2.imwrite(outputPath, upscaled)

    return (upscaled, end - start)


# imagePath = r"E:\dataset\satellite-data\frame-1000.jpg"
# image = cv2.imread(imagePath)
# print("[INFO] w: {}, h: {}".format(image.shape[1], image.shape[0]))
#
# upscaled, processtime = ESPCN_sResolution(image)
#
# print("[INFO] super resolution took {:.6f} seconds".format(processtime))
# print("[INFO] w: {}, h: {}".format(upscaled.shape[1], upscaled.shape[0]))
#
# cv2.imshow("Super Resolution", upscaled)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# run : python super-resolution.py -m "SR_models\EDSR_x4.pb" -i 'satellite-data\frame-1000.jpg'
