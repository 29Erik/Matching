# Imports
import cv2
import numpy as np
from matplotlib import pyplot as plot
import keyboard

#Parametros para la matriz de dispaeriedad
numDisparities = 16
blockSize = 15

#Parametro de control para las fotos
der = False
izq = False

#Lectura de la camara
cam = cv2.VideoCapture(0)
cv2.namedWindow("Imagen de la camara")
if (cam.isOpened() == False): 
  print("No podemos leer tu camara, LO SENTIMOS")

#Inicializacion de SIFT
sift = cv2.xfeatures2d.SIFT_create()

while(True):
    ret, original = cam.read()
    if ret == True:
        out = cv2.imwrite('Frame.jpg', original)
        cv2.imshow('Video Original',original)
        cv2.waitKey(1)
        if  keyboard.is_pressed('q'):
            break
        if keyboard.is_pressed('d'):
            ImageDer= cv2.imwrite('Derecha.jpg', original)
            ImageDer= cv2.imread('Derecha.jpg',0)
            der = True    
        if keyboard.is_pressed('i'):
            ImageIzq= cv2.imwrite('Izquierda.jpg', original)
            ImageIzq= cv2.imread('Izquierda.jpg',0)
            izq = True
        if der == True and izq == True:
            #Encuentra los puntos clave y descriptores con SIFT
            kp1, des1 = sift.detectAndCompute(ImageIzq,None)
            kp2, des2 = sift.detectAndCompute(ImageDer,None)

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

            def drawlines(ImageIzq,ImageDer,lines,pts1,pts2):
                ''' ImageIzq - image on which we draw the epilines for the points in ImageDer
                    lines - corresponding epilines '''
                r = ImageIzq.shape[0]
                c = ImageIzq.shape[1]
                ImageIzq = cv2.cvtColor(ImageIzq,cv2.COLOR_GRAY2BGR)
                ImageDer = cv2.cvtColor(ImageDer,cv2.COLOR_GRAY2BGR)
                for r,pt1,pt2 in zip(lines,pts1,pts2):
                    color = tuple(np.random.randint(0,255,3).tolist())
                    x0,y0 = map(int, [0, -r[2]/r[1] ])
                    x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
                    ImageIzq = cv2.line(ImageIzq, (x0,y0), (x1,y1), color,1)
                    ImageIzq = cv2.circle(ImageIzq,tuple(pt1),5,color,-1)
                    ImageDer = cv2.circle(ImageDer,tuple(pt2),5,color,-1)
                return ImageIzq,ImageDer

            #Encuentra epilíneas correspondientes a puntos en la imagen derecha (segunda imagen) y
            # dibujando sus líneas en la imagen de la izquierda
            lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
            lines1 = lines1.reshape(-1,3)
            img5,img6 = drawlines(ImageIzq,ImageDer,lines1,pts1,pts2)

            # Encuentra epilíneas correspondientes a puntos en la imagen de la izquierda (primera imagen) y
            # dibujando sus líneas en la imagen derecha
            lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
            lines2 = lines2.reshape(-1,3)
            img3,img4 = drawlines(ImageDer,ImageIzq,lines2,pts2,pts1)

            #Matriz de dispaeriedad
            stereo = cv2.StereoBM_create(numDisparities, blockSize)
            disparity = stereo.compute(ImageIzq,ImageDer)

            # Display images
            plot.subplot(211), plot.imshow(img5),plot.title("Lineas epipolares Izq.")
            plot.xticks([]), plot.yticks([])
            plot.subplot(212), plot.imshow(img3),plot.title("Lineas epipolares Der.")
            plot.xticks([]), plot.yticks([])
            plot.subplot(213), plot.imshow(disparity),plot.title("Mapa de dispariedad")
            plot.xticks([]), plot.yticks([])
            plot.show()
    else:
        break 
cam.release()
cv2.destroyAllWindows()
