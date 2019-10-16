import cv2
import numpy as np
from matplotlib import pyplot as plt
cap = cv2.VideoCapture(0)
 
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

 
while(True):
  ret, original = cap.read()
 
  if ret == True: 
    out = cv2.imwrite('Frame.jpg', original)  
    cv2.imshow('Video Original',original)
    cv2.waitKey(1)
    if  0xFF == ord('q'):
      break
    if 0xFF == 256:
        # SPACE pressed
        img_name = "Referencia.jpg".format(img_counter)
        imgRef=cv2.imwrite(img_name, out)
        print("Imagen guardada.".format(img_name))
    
    img_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(imgRef,0)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
      cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    cv2.imshow('Resultado',img_rgb)
    cv2.waitKey(1)
  else:
    break 
cap.release()
cv2.destroyAllWindows() 