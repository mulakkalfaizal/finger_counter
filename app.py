import os
import cv2
import mediapipe as mp
import time
import handTrackingModule as htm

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

folder_path = "finger_images"
myList = os.listdir(folder_path)
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f"{folder_path}/{imPath}")
    overlayList.append(image)

print(len(overlayList))

detector = htm.handDetector(detectionCon=0.75)

while True:
    success, img = cap.read()
    img = detector.findHands(img)

    h, w, c = overlayList[0].shape
    img[0:h, 0:w] = overlayList[0]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("image", img)
    cv2.waitKey(1)
