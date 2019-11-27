# Imports
import cv2
import numpy as np
from matplotlib import pyplot as plot

#Parametros para la matriz de dispaeriedad
numDisparities = 16
blockSize = 15

# Lectura de imagenes
img1 = cv2.imread('left.jpg',0) #queryimage # left image
img2 = cv2.imread('right.jpg',0) #trainimage # right image

#Inicializacion de SIFT
sift = cv2.xfeatures2d.SIFT_create()

#Encuentra los puntos clave y descriptores con SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parametros
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
good = []
pts1 = []
pts2 = []

# prueba de proporción según el paper de Lowe' s
for i,(m,n) in enumerate(matches):
  if m.distance < 0.8*n.distance:
    good.append(m)
    pts2.append(kp2[m.trainIdx].pt)
    pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

def drawlines(img1,img2,lines,pts1,pts2):
  ''' img1 - image on which we draw the epilines for the points in img2
      lines - corresponding epilines '''
  r,c = img1.shape
  img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
  img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
  for r,pt1,pt2 in zip(lines,pts1,pts2):
    color = tuple(np.random.randint(0,255,3).tolist())
    x0,y0 = map(int, [0, -r[2]/r[1] ])
    x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
    img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
    img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
    img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
  return img1,img2

#Encuentra epilíneas correspondientes a puntos en la imagen derecha (segunda imagen) y
# dibujando sus líneas en la imagen de la izquierda
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# Encuentra epilíneas correspondientes a puntos en la imagen de la izquierda (primera imagen) y
# dibujando sus líneas en la imagen derecha
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

#Matriz de dispariedad
stereo = cv2.StereoBM_create(numDisparities, blockSize)
disparity = stereo.compute(img1,img2)

# Display images
plot.subplot(131), plot.imshow(img5),plot.title("Lineas epipolares Izq.")
plot.xticks([]), plot.yticks([])
plot.subplot(132), plot.imshow(img3),plot.title("Lineas epipolares Der.")
plot.xticks([]), plot.yticks([])
plot.subplot(133), plot.imshow(disparity),plot.title("Mapa de dispariedad")
plot.xticks([]), plot.yticks([])
plot.show()
