# Basic Operation Python

# Accessing and Modifying pixel values
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import sys
from skimage import io
from scipy import ndimage as nd


def changePixel():
    img = cv2.imread(os.path.dirname(__file__) + '/test.png')

    print(img.size)

    px = img[0, 0]

    img.itemset((0, 0, 0), 220)
    img.itemset((0, 0, 1), 222)
    img.itemset((0, 0, 2), 223)

    print(px)
    plt.imshow([[px]], 'gray')
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


#changePixel()


def applyGuassianFilter():
    imgOrg = io.imread(os.path.dirname(__file__) + '/sample.jpg')
    gaussian_filter = nd.gaussian_filter(imgOrg, 3)
    median_filter = nd.median_filter(imgOrg, 10)

    fig = plt.figure(figsize=(4, 4))
    columns = 2
    rows = 2
    for i in range(1, columns*rows + 1):
        if (i == 1):
            img = imgOrg
        elif (i == 2):
            img = gaussian_filter
        else:
            img = median_filter
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)

    plt.show()


# applyGuassianFilter()

def threshold():
    img = cv2.imread(os.path.dirname(__file__) + '/sample.jpg', 0)
    # img = cv2.medianBlur(img,5)
    th = 127
    maxTh = 255
    ret, thresh1 = cv2.threshold(img, th, maxTh, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, th, maxTh, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, th, maxTh, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, th, maxTh, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, th, maxTh, cv2.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY',
            'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    folder = "threshold_output"

    for i in range(6):
        plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
        cv2.imwrite(os.path.dirname(__file__) + '/' + folder + '/' + titles[i] + ".jpg", images[i]) 

    plt.show()

def otsuThreshold():
    img = cv2.imread(os.path.dirname(__file__) + '/sample.jpg', 0)
    # img = cv2.medianBlur(img,5)
    # img = cv2.GaussianBlur(img,(5,5),0)
    th = 127
    maxTh = 255
    ret, thresh1 = cv2.threshold(img, th, maxTh, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, thresh2 = cv2.threshold(img, th, maxTh, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, thresh3 = cv2.threshold(img, th, maxTh, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
    ret, thresh4 = cv2.threshold(img, th, maxTh, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    ret, thresh5 = cv2.threshold(img, th, maxTh, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
    

    titles = ['Original Image', 'BINARY',
            'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    folder = "threshold_output"
    prefix = "OTSU_"

    for i in range(6):
        plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
        cv2.imwrite(os.path.dirname(__file__) + '/' + folder + '/' + prefix + titles[i] + ".jpg", images[i]) 

    plt.show()

# otsuThreshold()    
# exit()


def smoothing():

    img = cv2.imread(os.path.dirname(__file__) + '/sample.jpg', 0)

    # img = cv2.medianBlur(img,5)
    # img = cv2.GaussianBlur(img,(5,5),0)
    th = 127
    maxTh = 255
    ret, thresh1 = cv2.threshold(img, th, maxTh, cv2.THRESH_BINARY )
    kernel = np.ones((5,5),np.float32)/25
    smoothedImg = cv2.filter2D(thresh1,-1,kernel)
    

    titles = ['Original Image', 'BINARY',
            'smoothedImg']
    images = [img, thresh1, smoothedImg]

    folder = "threshold_output"
    prefix = "smooth_"

    for i in range(3):
        plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
        cv2.imwrite(os.path.dirname(__file__) + '/' + folder + '/' + prefix + titles[i] + ".jpg", images[i]) 

    plt.show()

# smoothing()    
# exit()


def morphologicalOps():

    img = cv2.imread(os.path.dirname(__file__) + '/sample.jpg', 0)

    # img = cv2.medianBlur(img,5)
    # img = cv2.GaussianBlur(img,(5,5),0)
    th = 127
    maxTh = 255
    ret, thresh1 = cv2.threshold(img, th, maxTh, cv2.THRESH_BINARY )
    kernel = np.ones((5,5),np.float32)/25
    smoothedImg = cv2.filter2D(thresh1,-1,kernel)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(smoothedImg,kernel,iterations = 1)
    dilation = cv2.dilate(smoothedImg,kernel,iterations = 1)
    opening = cv2.morphologyEx(smoothedImg, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(smoothedImg, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(smoothedImg, cv2.MORPH_GRADIENT, kernel)
    tophat = cv2.morphologyEx(smoothedImg, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(smoothedImg, cv2.MORPH_BLACKHAT, kernel)
    dilationAfterErosion = cv2.dilate(erosion,kernel,iterations = 1)
    edges = cv2.Canny(dilationAfterErosion, 100,200)
    
    image, contours, hierarchy = cv2.findContours(dilationAfterErosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cntImg = cv2.drawContours(img, contours, -1, (0,255,0), 3)

    titles = ['Original Image', 'smoothedImg',
            'erosion', 'dilation','opening', 'closing', 'gradient', 'tophat', 'blackhat', 'dilationAfterErosion']
    images = [img, smoothedImg, erosion, dilation, opening, closing, gradient, tophat, blackhat, dilationAfterErosion]

    folder = "threshold_output"
    prefix = "morph_"
    # x = cv2.resize(unknown, (960, 600))     
    # cv2.imshow("Image", x)
    # # cv2.imwrite(os.path.dirname(__file__) + '/' + folder + '/' + prefix + "contours" + ".jpg", cntImg) 
    # cv2.waitKey(0)
    # return

    for i in range(len(images)):
        plt.subplot(4, 3, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
        # cv2.imwrite(os.path.dirname(__file__) + '/' + folder + '/' + prefix + titles[i] + ".jpg", images[i]) 

    plt.show()

morphologicalOps()    
exit()


