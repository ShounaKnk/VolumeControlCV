import cv2
import mediapipe as mp
import time
import handTracking as htm
import math
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

prevTime =0
currTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector(detectionConf=0.8)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()

minVolume = volRange[0]
maxVolume = volRange[1]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) !=0:
        print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cv2.circle(img, (x1,y1), 5, (255, 0 ,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 5, (255, 0 ,255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2,y2), (255, 0 ,255), 3)

        length = math.hypot(x2-x1, y2-y1)
        print(length)

        vol = np.interp(length, [20, 150], [minVolume, maxVolume])
        volume.SetMasterVolumeLevel(vol, None)
        print(vol)
    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime=currTime
    cv2.putText(img, str(int(fps)), (10,50), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)
    cv2.imshow("image", img)
    cv2.waitKey(1)
