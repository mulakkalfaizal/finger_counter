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
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)

    if len(lmList) != 0:
        '''
        Lower value of y means open
        8 is the tip of index finger
        
        if lmList[8][2] < lmList[6][2]:
            print("Index Finger Open")
        else:
            print("index finger closed")
        '''
        fingers = []
        '''
        Logic for thumb finger
        for thump the tip will not go below the second point so we have depend on hte x axis,
        ie if the tip of the thump is moved to right compared to below tip (-1) then it is considered as folded
        '''
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        '''
        Logic for other Four fingers
        if the tip of the four fingers are below the second tip (-2 as per the landmark diagram) then it is 
        considered as folded. 
        '''
        for id in range(1, 5):
            # Checking tip point (y) is less than tip -2 point (y) then it is open
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers)

    h, w, c = overlayList[0].shape
    img[0:h, 0:w] = overlayList[0]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("image", img)
    cv2.waitKey(1)
