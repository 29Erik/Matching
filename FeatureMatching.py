import cv2
import numpy as np
import keyboard
from matplotlib import pyplot as plt
cap = cv2.VideoCapture(0)
MIN_MATCH_COUNT = 4

orb = cv2.ORB_create()
 
if (cap.isOpened() == False): 
  print("No podemos leer tu camara, LO SENTIMOS")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

finded = False
cropping = False
cropped = False
recorte1= False
recorte2= False
recorte3= False
recorte4 = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
oriImage= None
templateFull= None
templateFull2= None
templateFull3= None
templateFull4= None
contObj = 0

def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping, cropped, oriImage, templateFull, templateFull2, templateFull3, templateFull4, contObj, recorte1, recorte2, recorte3, recorte4
 
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
            contObj += 1
            if recorte1 == True:
                templateFull = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                cv2.destroyWindow('Objeto 1')
                img_name = "Cropped_{}".format(contObj)
                cv2.imshow(img_name, templateFull)
            if recorte2 == True:
                templateFull2 = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                cv2.destroyWindow('Objeto 2')
                img_name = "Cropped_{}".format(contObj)
                cv2.imshow(img_name, templateFull2)
            if recorte3 == True:
                templateFull3 = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                cv2.destroyWindow('Objeto 3')
                img_name = "Cropped_{}".format(contObj)
                cv2.imshow(img_name, templateFull3)
            if recorte4 == True:
                templateFull4 = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                cv2.destroyWindow('Objeto 4')
                img_name = "Cropped_{}".format(contObj)
                cv2.imshow(img_name, templateFull4)
            recorte1= False
            recorte2= False
            recorte3= False
            recorte4 = False
            if contObj >= 4:
                cropped=True

def Matching(imgOriginal, imgCaptura):
    kptsOriginal, descsOriginal = orb.detectAndCompute(imgOriginal,None)
    kptsCaptura, descsCaptura = orb.detectAndCompute(imgCaptura,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descsOriginal, descsCaptura)
    dmatches = sorted(matches, key = lambda x:x.distance)

    src_pts  = np.float32([kptsOriginal[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
    dst_pts  = np.float32([kptsCaptura[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    h,w = imgOriginal.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    return dst
 
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
        template = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        oriImage= template.copy()
        recorte1 = True        

    if recorte1 == True:
        w, h = template.shape[::-1]
        cv2.imshow('Objeto 1',template)
        cv2.setMouseCallback("Objeto 1", mouse_crop)
        i = template.copy()
    
        if not cropping:
            cv2.imshow("Objeto 1", template)
        
        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("Objeto 1", i)
        
        cv2.waitKey(1)

    if keyboard.is_pressed('2'):
        template2 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        oriImage= template2.copy()
        recorte2 = True 

    if recorte2 == True:
        w, h = template2.shape[::-1]
        cv2.imshow('Objeto 2',template2)
        cv2.setMouseCallback("Objeto 2", mouse_crop)
        i = template2.copy()
    
        if not cropping:
            cv2.imshow("Objeto 2", template2)
        
        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("Objeto 2", i)
        
        cv2.waitKey(1)

    if keyboard.is_pressed('3'):
        template3 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        oriImage= template3.copy()
        recorte3 = True 
    
    if recorte3 == True:
        w, h = template3.shape[::-1]
        cv2.imshow('Objeto 3',template3)
        cv2.setMouseCallback("Objeto 3", mouse_crop)
        i = template3.copy()
    
        if not cropping:
            cv2.imshow("Objeto 3", template3)
        
        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("Objeto 3", i)
        
        cv2.waitKey(1)

    if keyboard.is_pressed('4'):
        template4 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        oriImage= template4.copy()
        recorte4 = True 
    
    if recorte4 == True:
        w, h = template4.shape[::-1]
        cv2.imshow('Objeto 4',template4)
        cv2.setMouseCallback("Objeto 4", mouse_crop)
        i = template4.copy()
    
        if not cropping:
            cv2.imshow("Objeto 4", template4)
        
        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("Objeto 4", i)
        
        cv2.waitKey(1)

    if contObj >= 4:
        finded = True


    if finded:
      cv2.destroyWindow('Video Original')
      
      img_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
      img_rgb= img_gray.copy()
      img_rgb2 = img_gray.copy()
      img_rgb3 = img_gray.copy()
      img_rgb4 = img_gray.copy()
            
      if cropped:       
        resultado1 = Matching(img_rgb, templateFull)
        resultado2 = Matching(img_rgb2, templateFull2)
        resultado3 = Matching(img_rgb3, templateFull3)
        resultado4 = Matching(img_rgb4, templateFull4)

        frame1 = cv2.polylines(img_rgb, [np.int32(resultado1)], True, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow("ENCONTRADO MATCHING CON OBJ 1", frame1)

        frame2 = cv2.polylines(img_rgb, [np.int32(resultado2)], True, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow("ENCONTRADO MATCHING CON OBJ 2", frame2)

        frame3 = cv2.polylines(img_rgb, [np.int32(resultado3)], True, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow("ENCONTRADO MATCHING CON OBJ 3", frame3)

        frame4 = cv2.polylines(img_rgb, [np.int32(resultado4)], True, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow("ENCONTRADO MATCHING CON OBJ 4", frame4)
      
      cv2.waitKey(1)
  else:
    break 
cap.release()
cv2.destroyAllWindows() 