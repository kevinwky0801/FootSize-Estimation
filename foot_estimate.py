import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from sklearn.cluster import KMeans
import random as random

def preprocess(img, img_class):
    img = enhance_contrast(img)  # New: enhance contrast using CLAHE

    if img_class in [2, 3]:
        gamma_img = gammaCorrection(img, 2)
        gamma_img = cv2.medianBlur(gamma_img, 21)
        hsv = cv2.cvtColor(gamma_img, cv2.COLOR_RGB2HSV_FULL)
        if img_class == 2:
            lower = np.array([27, 0, 0])
            upper = np.array([155, 255, 255])
        if img_class == 3:
            lower = np.array([0, 23, 0])
            upper = np.array([179, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        img = cv2.bitwise_and(gamma_img, gamma_img, mask=mask)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = cv2.medianBlur(img, 21)
    img = img / 255
    return img

def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def plotImage(img, title):
    plt.imshow(img)

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)

def cropOrig(bRect, oimg):
    x, y, w, h = bRect
    pcropedImg = oimg[y:y + h, x:x + w]
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]
    y2 = int(h1 / 10)
    x2 = int(w1 / 10)
    crop1 = pcropedImg[y1 + y2:h1 - y2, x1 + x2:w1 - x2]
    ix, iy, iw, ih = x + x2, y + y2, crop1.shape[1], crop1.shape[0]
    croppedImg = oimg[iy:iy + ih, ix:ix + iw]
    return croppedImg, pcropedImg

def overlayImage(croppedImg, pcropedImg):
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]
    y2 = int(h1 / 10)
    x2 = int(w1 / 10)
    new_image = np.zeros((pcropedImg.shape[0], pcropedImg.shape[1], 3), np.uint8)
    new_image[:, 0:pcropedImg.shape[1]] = (117, 13, 205)
    new_image[y1 + y2:y1 + y2 + croppedImg.shape[0], x1 + x2:x1 + x2 + croppedImg.shape[1]] = croppedImg
    new_image[np.where((new_image == [246, 57, 178]).all(axis=2))] = [117, 13, 205]
    new_image[np.where((new_image == [57, 6, 180]).all(axis=2))] = [117, 13, 205]
    new_img = cv2.medianBlur(new_image, 17)
    return new_img

def kMeans_cluster(img, img_class):
    image_2D = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    n_clusters = 3 if img_class in [1, 2, 3, 4] else 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_2D)
    clustOut = kmeans.cluster_centers_[kmeans.labels_]
    clustered_3D = clustOut.reshape(img.shape[0], img.shape[1], img.shape[2])
    clusteredImg = np.uint8(clustered_3D * 255)
    return clusteredImg

def getBoundingBox(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    return boundRect, contours, contours_poly, img

def drawCnt(bRect, contours, cntPoly, img):
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    paperbb = bRect
    for i in range(len(contours)):
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        cv2.drawContours(drawing, cntPoly, i, color)
    cv2.rectangle(drawing, (int(paperbb[0]), int(paperbb[1])),
                  (int(paperbb[0] + paperbb[2]), int(paperbb[1] + paperbb[3])), color, 2)
    return drawing

def paperEdgeDetection(clusteredImage):
    edged1 = cv2.Canny(clusteredImage, 0, 255, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.dilate(edged1, kernel, iterations=10)
    edged = cv2.erode(edged, kernel, iterations=8)
    edged = cv2.fastNlMeansDenoising(edged, None, 20, 7, 21)
    return edged

def edgeDetection(clusteredImage):
    edged1 = cv2.Canny(clusteredImage, 0, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.dilate(edged1, kernel, iterations=10)
    edged = cv2.erode(edged, kernel, iterations=9)
    edged = cv2.fastNlMeansDenoising(edged, None, 20, 7, 21)
    return edged

def footEdgeDetection(clusteredImage):
    edged1 = cv2.Canny(clusteredImage, 0, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edged = cv2.dilate(edged1, kernel, iterations=2)
    edged = cv2.erode(edged, kernel, iterations=2)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)
    return edged

def calcFeetSize(pcropedImg, fboundRect):
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]
    y2 = int(h1 / 10)
    x2 = int(w1 / 10)
    fh = y2 + fboundRect[0][3]
    fw = x2 + fboundRect[0][2]
    ph = pcropedImg.shape[0]
    pw = pcropedImg.shape[1]
    opw = 210
    oph = 297
    ofs_w = opw / pw * fw
    ofs_h = oph / ph * fh
    return ofs_w / 10, ofs_h / 10, fh, fw, ph, pw
