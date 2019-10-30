import numpy as np
import cv2

imgname = "objeto.png"
cap = cv2.VideoCapture(0)
MIN_MATCH_COUNT = 4

orb = cv2.ORB_create()
img1 = cv2.imread(imgname)
while(True):
    ret, frame = cap.read()

    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    kpts1, descs1 = orb.detectAndCompute(gray1,None)
    kpts2, descs2 = orb.detectAndCompute(gray2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descs1, descs2)
    dmatches = sorted(matches, key = lambda x:x.distance)

    src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
    dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    h,w = img1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    frame = cv2.polylines(frame, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
    cv2.imshow("found", frame)

    res = cv2.drawMatches(img1, kpts1, frame, kpts2, dmatches[:20],None,flags=2)
    #cv2.imshow("RESULTADO", res)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()