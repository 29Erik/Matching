import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('control-template.jpg',0)
img2 = cv2.imread('control-image.jpg',0)

brisk = cv2.BRISK_create()

kp1, des1 = brisk.detectAndCompute(img1,None)
kp2, des2 = brisk.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
plt.imshow(img3)
plt.show()