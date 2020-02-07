import numpy as np
import cv2
from pathlib import Path

im = cv2.imread('images/sample_3.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

k = cv2.drawContours(thresh, contours, -1, (0,255,0), 3)

cv2.imshow("Contours", k)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(k)