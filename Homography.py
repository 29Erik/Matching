import cv2
import numpy as np
import matplotlib.pyplot as plot

if __name__ == '__main__' :
 
    # Read source image.
    im_src = cv2.imread('book2.jpg')
    # Four corners of the book in source image
    pts_src = np.array([[141, 131], [480, 159], [493, 630],[64, 601]])
 
 
    # Read destination image.
    im_dst = cv2.imread('book1.jpg')
    # Four corners of the book in destination image.
    pts_dst = np.array([[318, 256],[534, 372],[316, 670],[73, 473]])
 
    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)
     
    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
     
    # Display images
    plot.subplot(131), plot.imshow(im_src),plot.title("Imagen origen")
    plot.xticks([]), plot.yticks([])
    plot.subplot(132), plot.imshow(im_dst),plot.title("Imagen destino")
    plot.xticks([]), plot.yticks([])
    plot.subplot(133), plot.imshow(im_out),plot.title("Imagen transformada")
    plot.xticks([]), plot.yticks([])
    plot.show()
    cv2.waitKey(0)