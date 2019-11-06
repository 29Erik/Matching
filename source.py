import cv2
import numpy as np
import keyboard
from matplotlib import pyplot as plt
cap = cv2.VideoCapture(0)
 
if (cap.isOpened() == False): 
  print("No podemos leer tu camara, LO SENTIMOS")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

finded = False
cropping = False
cropped = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
oriImage= None
templateFull= None

def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping, cropped, oriImage, templateFull
 
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
        cropped=False
 
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
 
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
 
        refPoint = [(x_start, y_start), (x_end, y_end)]
 
        if len(refPoint) == 2: #when two points were found
            templateFull = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", templateFull)
            cropped=True
 
while(True):
  ret, original = cap.read()
 
  if ret == True: 
    out = cv2.imwrite('Frame.jpg', original)
    if finded == False:
        cv2.imshow('Video Original',original)
        cv2.waitKey(1)
    if  keyboard.is_pressed('q'):
      break
    if keyboard.is_pressed('1'):
        # SPACE pressed
        template = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        print("TOMO CAPTURA")
        finded = True
    if finded:
      cv2.destroyWindow('Video Original')
      oriImage= template.copy()
      img_rgb= original.copy()
      img_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
      w, h = template.shape[::-1]
      cv2.imshow('Captura',template)
      cv2.setMouseCallback("Captura", mouse_crop)
      i = template.copy()
 
      if not cropping:
        cv2.imshow("Captura", template)
    
      elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("Captura", i)
    
      cv2.waitKey(1)
      if cropped:
        res = cv2.matchTemplate(img_gray,templateFull,cv2.TM_CCOEFF_NORMED)
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