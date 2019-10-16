import cv2
import numpy as np
import keyboard
from matplotlib import pyplot as plt
cap = cv2.VideoCapture(0)
 
if (cap.isOpened() == False): 
  print("No podemos leer tu camara, LO SENTIMOS")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

 
while(True):
  ret, original = cap.read()
 
  if ret == True: 
    out = cv2.imwrite('Frame.jpg', original)  
    cv2.imshow('Video Original',original)
    cv2.waitKey(1)
    finded = False
    if  keyboard.is_pressed('q'):
      break
    if keyboard.is_pressed('s'):
        # SPACE pressed
        imgRef=cv2.imwrite("Referencia.jpg", out)
        finded = True
    if finded:
      img_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
      template = cv2.imread(imgRef,0)
      template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
      w, h = template.shape[::-1]

      res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
      threshold = 0.8
      loc = np.where( res >= threshold)
      for pt in zip(*loc[::-1]):
        cv2.rectangle(out, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
      cv2.imshow('Resultado',out)
      cv2.waitKey(1)
  else:
    break 
cap.release()
cv2.destroyAllWindows() 