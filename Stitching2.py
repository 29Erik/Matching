import cv2
import numpy as np
import matplotlib.pyplot as plt

ImagenDerecha = cv2.imread('right1.jpg')
img1 = cv2.cvtColor(ImagenDerecha,cv2.COLOR_BGR2GRAY)
ImagenIzquierda = cv2.imread('left1.jpg')
img2 = cv2.cvtColor(ImagenIzquierda,cv2.COLOR_BGR2GRAY)
#############################################################################
plt.subplot(231),plt.imshow(ImagenDerecha),plt.title('Original Derecha')
plt.subplot(232),plt.imshow(ImagenIzquierda),plt.title('Original Izquierda')
#############################################################################

sift = cv2.xfeatures2d.SIFT_create()
# find key points
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
PtsKey = cv2.drawKeypoints(ImagenDerecha,kp1,None)
#############################################################################
plt.subplot(233),plt.imshow(PtsKey),plt.title('Pts clave Der.')
#############################################################################
#FLANN_INDEX_KDTREE = 0
match = cv2.BFMatcher()
matches = match.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.03*n.distance:
        good.append(m)

draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=None,
                       flags=2)

MAtches = cv2.drawMatches(ImagenDerecha,kp1,ImagenIzquierda,kp2,good,None,**draw_params)
#############################################################################
plt.subplot(234),plt.imshow(MAtches),plt.title('Pts clave Imagen.')
#############################################################################
MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    Intersect = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    #############################################################################
    plt.subplot(235),plt.imshow(Intersect),plt.title('Linea de interseccion.')
    #############################################################################
else:
    print("No se encuentra suficientes puntos de match - %d/%d", (len(good)/MIN_MATCH_COUNT))

dst = cv2.warpPerspective(ImagenDerecha,M,(ImagenIzquierda.shape[1] + ImagenDerecha.shape[1], ImagenIzquierda.shape[0]))
dst[0:ImagenIzquierda.shape[0],0:ImagenIzquierda.shape[1]] = ImagenIzquierda

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top

    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

#############################################################################
plt.subplot(236),plt.imshow(trim(dst)),plt.title('Imagen en mosaico')
cv2.imwrite("IMAGEN MOISAQUEADA.jpg", trim(dst))
plt.show()
#############################################################################