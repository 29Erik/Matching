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
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  else:
    break 
cap.release()
cv2.destroyAllWindows() 