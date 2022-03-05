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
rawList = os.listdir(folder_path)
myList = sorted(rawList, key=lambda x: x[:1])

overlayList = []
for imPath in myList:
    image = cv2.imread(f"{folder_path}/{imPath}")
    overlayList.append(image)

detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)

    # This will give the full landmark of the fingers and the position
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)

    if len(lmList) != 0:
        fingers = []
        '''
        Logic for thumb finger
        for thump the tip will not go below the second point so we have depend on hte x axis,
        ie if the tip of the thump is moved to right compared to below tip (-1) then it is considered as folded
        lmList[tipIds[0]][1] is the x position of tip of thump
        lmList[tipIds[0] - 1][1] is the x postion of the below point of the thump
        '''
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        '''
        Logic for other Four fingers
        if the tip of the four fingers are below the second tip (tip -2 as per the landmark diagram) then it is
        considered as folded.
        '''
        for id in range(1, 5):
            # Checking tip point (y) is less than tip -2 point (y) then it is open
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Count the number of 1's in the list which gives us how many fingers are open
        totalFingers = fingers.count(1)
        print(f"Total finger open is {totalFingers}")

        # display the corresponding image based on the number of fingers open.
        h, w, c = overlayList[totalFingers - 1].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1]

        # display the text also mentioning how many fingers are  open
        cv2.rectangle(img, (20, 525), (170, 700), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (40, 660), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("image", img)
    cv2.waitKey(1)
