import cv2
import numpy as np
#Read the images from your directory
dim=(1024,768)
left=cv2.imread('left.jpg',cv2.IMREAD_COLOR)
left=cv2.resize(left,dim,interpolation = cv2.INTER_AREA)   #ReSize to (1024,768)
right=cv2.imread('right.jpg',cv2.IMREAD_COLOR)
right=cv2.resize(right,dim,interpolation = cv2.INTER_AREA) #ReSize to (1024,768)

images=[]
images.append(left)
images.append(right)


stitcher = cv2.createStitcher()
#stitcher = cv2.Stitcher.create()
ret,pano = stitcher.stitch(images)

if ret==cv2.STITCHER_OK:
    cv2.imshow('Panoramica',pano)
    cv2.imwrite('MOSAICO.jpg', pano)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print("Error en el Stitching")