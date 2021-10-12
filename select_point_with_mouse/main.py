import numpy as np
import cv2 as cv 
import AiPhile
import time
def selectPoint(event, x , y , flags, parmas):
    global point, condition
    if event ==cv.EVENT_LBUTTONUP:
        point =(int(x), int(y))
        condition =True 

cv.namedWindow('frame')
cv.setMouseCallback('frame', selectPoint)

cap = cv.VideoCapture(1)
frame_counter=0
point =()
condition = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_counter += 1
    fps = frame_counter/(time.perf_counter())
    AiPhile.textBGoutline(frame, f'FPS: {round(fps,2)}', (40,30), scaling=0.8)
    if condition:
        cv.circle(frame, point, 5, AiPhile.MAGENTA, -1, cv.LINE_AA)

    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
cap.release()