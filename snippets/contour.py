import numpy as np
import cv2
from pathlib import Path

im = cv2.imread('images/sample_3.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray, 127, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

cnts = cv2.drawContours(thresh, contours, -1, (0,255,0), 3)

# cv2.imshow("Contours", im2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

rect = cv2.minAreaRect(contours)
box = cv2.boxPoints(rect)


cv2.imshow('contours', cv2.drawContours(thresh,[box],0,(0,0,255),2))

for contours in cnts:
    print(cv2.boundingRect(contours))
    # cv2.imshow('img',thresh)
    # cv2.imshow('contours', cv2.boundingRect(contours))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 
    #  pointPolygonTest()


print("ssssssssssss", cnts)