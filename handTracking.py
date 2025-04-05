import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionConf,
            min_tracking_confidence=self.trackConf
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLndmks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLndmks, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw =True):
            landmakr_list = []
            if self.results.multi_hand_landmarks:
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c= img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    # print(f'{id}  {cx}  {cy}')
                    landmakr_list.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy),5, (255, 0, 0), cv2.FILLED)
            return landmakr_list


def main():
    prevTime =0
    currTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
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


if __name__ == "__main__":
    main()