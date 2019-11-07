import matplotlib.pyplot as plot
import numpy as np
import cv2
#Cargamos la imagen deseada
#img = cv2.imread("road.png")
img = cv2.imread("roads.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#Indicamos la escala a la que se reducirá.... 1/scale
scale = 4

#Escalamos la imagen
img_rs = cv2.resize(img, None, fx=1./scale, fy=1./scale, interpolation=cv2.INTER_LANCZOS4)

print ('Tamaño de imagen: \nimg: ')
print (img.shape)
print ('img_rs: ')
print (img_rs.shape)

#Mostramos la imagen original
plot.subplot(131), plot.imshow(img),plot.title("Original")
plot.xticks([]), plot.yticks([])
#Mostramos la imagen escalada
plot.subplot(132), plot.imshow(img_rs),plot.title("Escalada")
plot.xticks([]), plot.yticks([])

#Homografía
rows = img_rs.shape[0]; 
cols = img_rs.shape[1]

#Seleccionamos cuatro puntos usando un arreglo de numpy
pts1 = np.float32([[0,0],[0,rows],[cols,0],[cols,rows]])

#Dibujamos pts1 en la imagen escalada
for pts in pts1:
    circ = plot.Circle(pts,10)
#Seleccionamos cuatro putnos de destino
x = 280
pts2 = np.float32([[0,0],[x,rows],[cols,0],[cols-x,rows]])

#Se calcula la matriz para la corrección de perspectiva
M = cv2.getPerspectiveTransform(pts1,pts2)

#Obtenemos la imagen con corrección de pespectiva
img_hom = cv2.warpPerspective(img_rs, M, (cols,rows))

#Mostramos la imagen resultante
plot.subplot(133), plot.imshow(img_hom),plot.title("Salida")
plot.xticks([]), plot.yticks([])

#Dibujamos pts2
for pts in pts2:
    circ = plot.Circle(pts,10)

plot.show()