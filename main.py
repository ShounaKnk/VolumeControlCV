import cv2
import mediapipe as mp

mpHands = mp.solutions.hands
hands =  mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLndmks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLndmks, mpHands.HAND_CONNECTIONS)
    cv2.imshow("image", img)
    cv2.waitKey(1)