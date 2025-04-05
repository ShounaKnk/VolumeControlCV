import cv2
import mediapipe as mp
import time
import handTracking as ht
prevTime =0
currTime = 0
cap = cv2.VideoCapture(0)
detector = ht.handDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) !=0:
        print(lmList[4])
    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime=currTime
    cv2.putText(img, str(int(fps)), (10,50), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)
    cv2.imshow("image", img)
    cv2.waitKey(1)
